// ------------------------------------------------------------------
// MS-CNN
// Copyright (c) 2016 The Regents of the University of California
// see mscnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/detection_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DetectionSoftmaxForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const Dtype* bootstrap_map, const Dtype* weight_map,
          Dtype* loss, const int cls_dim, const int label_dim, const int spatial_dim, 
          const bool objectness, const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int base_index = n * spatial_dim + s;
    int label_value = static_cast<int>(label[n * label_dim + s]);
    if (objectness) {
      label_value = min(1,label_value);
    } 
    const int keep_flag = static_cast<int>(bootstrap_map[base_index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else if ((label_value == 0) && (keep_flag == 0)) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * cls_dim + label_value * spatial_dim + s], Dtype(FLT_MIN)))
                    * weight_map[base_index];
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void DetectionBoxForwardGPU(const int nthreads,
          const Dtype* coord_data, const Dtype* label, Dtype* coord_diff, const int coord_dim, 
          const int label_dim, const int spatial_dim, const int dr, const int height, 
          const int width, const int field_h, const int field_w, const float field_whr, 
          const float field_xyr, const int coord_num, const bool has_ignore_label, 
          const int ignore_label, Dtype* counts) {
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int h = s / width, w = s % width; 
    const int label_index = n * label_dim + s;
    const int label_value = static_cast<int>(label[label_index]);
    if ((has_ignore_label && label_value == ignore_label) || (label_value == 0)) {
      counts[index] = 0;
    } else {
      Dtype gx, gy, gw, gh; 
      gx = (label[label_index+spatial_dim]-(w+Dtype(0.5))*dr) / field_w;
      gy = (label[label_index+2*spatial_dim]-(h+Dtype(0.5))*dr) / field_h;
      gw = log(max(label[label_index+3*spatial_dim],Dtype(2)) / field_w);
      gh = log(max(label[label_index+4*spatial_dim],Dtype(2)) / field_h);

      // euclidean gradient
      Dtype tx, ty, tw, th;
      const int coord_idx = n * coord_dim + s;
      tx = coord_data[coord_idx];
      ty = coord_data[coord_idx+spatial_dim];
      tw = coord_data[coord_idx+2*spatial_dim];
      th = coord_data[coord_idx+3*spatial_dim];
      tx = max(min_xyr,tx); tx = min(max_xyr,tx); 
      ty = max(min_xyr,ty); ty = min(max_xyr,ty); 
      tw = max(min_whr,tw); tw = min(max_whr,tw); 
      th = max(min_whr,th); th = min(max_whr,th);

      coord_diff[coord_idx] = tx-gx;
      coord_diff[coord_idx+spatial_dim] = ty-gy;
      coord_diff[coord_idx+2*spatial_dim] = tw-gw;
      coord_diff[coord_idx+3*spatial_dim] = th-gh;
      counts[index] = coord_num;
    }
  }
}

template <typename Dtype>
__global__ void DetectionBBSmoothGPU(const int nthreads,
          Dtype* diff_data, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (diff_data[index] <= -1) { 
      loss[index] = (abs(diff_data[index])-Dtype(0.5));
      diff_data[index] = Dtype(-1);
    } else if (diff_data[index] < 1) {
      loss[index] = (diff_data[index]*diff_data[index]/Dtype(2));
    } else {
      loss[index] = (abs(diff_data[index])-Dtype(0.5));
      diff_data[index] = Dtype(1);
    }
  }
}
template <typename Dtype>
void DetectionLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int num = prob_.num();
  const int spatial_dim = prob_.height() * prob_.width();
  const int bottom_dim = bottom[0]->count() / num;
  const int coord_dim = coord_bottom_.count() / num;
  const int cls_dim = prob_.count() / num;
  const int label_dim = bottom[1]->count() / num;
  const int nthreads = num * spatial_dim;

  Dtype* cls_bottom_data = cls_bottom_.mutable_gpu_data();
  Dtype* coord_bottom_data = coord_bottom_.mutable_gpu_data();

  //build the bootstrap map
  const Dtype* label_cpu = bottom[1]->cpu_data();
  const Dtype* bottom_data_cpu = bottom[0]->cpu_data();
  Dtype* bootstrap_map_cpu = bootstrap_map_.mutable_cpu_data();
  caffe_set(bootstrap_map_.count(), Dtype(0), bootstrap_map_cpu);
  int total_neg_num = 0, total_pos_num = 0, positive_num, keep_num = 0;
  bool sample_all_flag = true;

  for (int i = 0; i < num; ++i) {
    vector<int> instance_nums(cls_num_);
    positive_num = 0;
    for (int c = 0; c < cls_num_; c++) instance_nums[c] = 0;
    for (int j = 0; j < spatial_dim; j++) {
      int base_index = i*label_dim+j;
      int label_value = static_cast<int>(label_cpu[base_index]);
      if (objectness_) {
        label_value = std::min(1,label_value);
      } 
      instance_nums[label_value]++;
      if (label_value != 0) positive_num++; 
    }
    if ((sample_mode_ == "random") || (sample_mode_ == "mixture")) {
      float bg_ratio = 1.0f; sample_all_flag = false;
      if (sample_mode_ == "mixture") bg_ratio = 0.5f; 
      int rand_sample_num = positive_num*bg_multiple_*bg_ratio;
      rand_sample_num = std::max(rand_sample_num, 4*(cls_num_-1));
      for (int nn = 0; nn < rand_sample_num; nn++) {
        int idx  = caffe_rng_rand() % spatial_dim;
        int base_index = i*label_dim+idx;
        const int label_value = static_cast<int>(label_cpu[base_index]);
        if ((label_value==0) && (label_cpu[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
          if (bootstrap_map_cpu[i*spatial_dim+idx] == 0) {keep_num++;}
          bootstrap_map_cpu[i*spatial_dim+idx] = 1;
        }
      }
    } 
    if ((sample_mode_ == "bootstrap") || (sample_mode_ == "mixture")) {
      float bg_ratio = 1.0f; sample_all_flag = false;
      if (sample_mode_ == "mixture") bg_ratio = 0.5f;
      for (int k = 1; k < cls_num_; ++k) {
        std::vector<std::pair<Dtype, int> > loc_vector;
        for (int j = 0; j < spatial_dim; j++) {
          int base_index = i*label_dim+j;
          const int label_value = static_cast<int>(label_cpu[base_index]);
          if ((label_value==0) && (label_cpu[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
            loc_vector.push_back(std::make_pair(bottom_data_cpu[i*bottom_dim+j+k*spatial_dim], j));
          }
        }
        int sort_num = instance_nums[k]*bg_multiple_*bg_ratio;
        sort_num = std::max(4, sort_num);
        sort_num =  std::min(int(loc_vector.size()), sort_num);
        if (sort_num == 0) continue;
        std::partial_sort(loc_vector.begin(), loc_vector.begin() + sort_num,
             loc_vector.end(), std::greater<std::pair<Dtype, int> >());
        for (int nn = 0; nn < sort_num; nn++) {
          int idx = loc_vector[nn].second;
          if (bootstrap_map_cpu[i*spatial_dim+idx] == 0) {keep_num++;}
          bootstrap_map_cpu[i*spatial_dim+idx] = 1;
        }
      }
    } 
    if (sample_all_flag) { 
      for (int j = 0; j < spatial_dim; j++) {
        int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label_cpu[base_index]);
        if ((label_value==0) && (label_cpu[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
          if (bootstrap_map_cpu[i*spatial_dim+j] == 0) {keep_num++;}
          bootstrap_map_cpu[i*spatial_dim+j] = 1;
        }
      }
    }
    total_neg_num += instance_nums[0]; total_pos_num += positive_num;
  }
  int keep_num_check = caffe_cpu_asum(num*spatial_dim,bootstrap_map_cpu);
  CHECK_EQ(keep_num, keep_num_check);
  DLOG(INFO)<<"cls num = "<<cls_num_<<", total positive = "<<total_pos_num
            <<", total negative = "<<total_neg_num<<", keep = "<<keep_num;

  // setup weight map
  Dtype* weight_map_cpu = weight_map_.mutable_cpu_data();
  caffe_set(weight_map_.count(), Dtype(1), weight_map_cpu);
  if (iou_weighted_) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label_cpu[base_index]);
        if (label_value != 0) {
          weight_map_cpu[i*spatial_dim+j] = label_cpu[base_index+spatial_dim+coord_dim];
        }
      }
    }
  }

  Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
  if (pos_neg_weighted_) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label_cpu[base_index]);
        if (label_value != 0) {
          pos_weight_sum += weight_map_cpu[i*spatial_dim+j]; 
        } else if (bootstrap_map_cpu[i*spatial_dim+j] == 1) {
          neg_weight_sum += weight_map_cpu[i*spatial_dim+j]; 
        }
      }
    }
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label_cpu[base_index]);
        Dtype fg_weight = Dtype(1)/(1+bg_multiple_);
        int sample_num = keep_num+total_pos_num;
        if (label_value != 0) {
          if (pos_weight_sum != 0) {
            weight_map_cpu[i*spatial_dim+j] *= (fg_weight*sample_num/Dtype(pos_weight_sum)); 
          }
        } else {
          if (neg_weight_sum != 0) {
            weight_map_cpu[i*spatial_dim+j] *= ((1-fg_weight)*sample_num/Dtype(neg_weight_sum)); 
          }
        }
      }
    }
  }

  for (int i = 0; i < num; i++) {
    caffe_gpu_memcpy(cls_dim * sizeof(Dtype), bottom_data+i*bottom_dim, cls_bottom_data+i*cls_dim);
    caffe_gpu_memcpy(coord_dim * sizeof(Dtype), bottom_data+i*bottom_dim+cls_dim, 
                     coord_bottom_data+i*coord_dim);
  }
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* cls_loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* cls_counts = prob_.mutable_gpu_diff();
  const Dtype* bootstrap_map = bootstrap_map_.gpu_data();
  const Dtype* weight_map = weight_map_.gpu_data();
  DetectionSoftmaxForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, bootstrap_map, weight_map, cls_loss_data, 
      cls_dim, label_dim, spatial_dim, objectness_, has_ignore_label_, ignore_label_, cls_counts);

  Dtype cls_loss;
  caffe_gpu_asum(nthreads, cls_loss_data, &cls_loss);

  // the forward pass computes euclidean loss
  const int label_height = bottom[1]->height(), label_width = bottom[1]->width();
  Dtype* coord_diff_data = coord_diff_.mutable_gpu_data();
  caffe_gpu_set(coord_diff_.count(), Dtype(0), coord_diff_data);
  Dtype* coord_counts = coord_diff_.mutable_gpu_diff();
  DetectionBoxForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, coord_bottom_data, label, coord_diff_data,
      coord_dim, label_dim, spatial_dim, downsample_rate_, label_height, label_width, 
      field_h_, field_w_, field_whr_, field_xyr_, coord_num_, has_ignore_label_, 
      ignore_label_, coord_counts);

  Dtype coord_loss;
  if (bb_smooth_) {
    Dtype* coord_loss_data = bottom[0]->mutable_gpu_diff();
    DetectionBBSmoothGPU<Dtype><<<CAFFE_GET_BLOCKS(coord_diff_.count()),
        CAFFE_CUDA_NUM_THREADS>>>(coord_diff_.count(), coord_diff_data, coord_loss_data);
    caffe_gpu_asum(coord_diff_.count(),coord_loss_data, &coord_loss);
  } else {
    caffe_gpu_dot(coord_diff_.count(), coord_diff_.gpu_data(), coord_diff_.gpu_data(), &coord_loss);
    coord_loss /= Dtype(2);
  }

  Dtype cls_count, coord_count;
  caffe_gpu_asum(nthreads, cls_counts, &cls_count); 
  caffe_gpu_asum(nthreads, coord_counts, &coord_count);
  if (cls_count == 0) cls_loss = 0;
  else cls_loss /= cls_count;
  if (coord_count == 0) coord_loss = 0;
  else coord_loss /= coord_count;
  
  top[0]->mutable_cpu_data()[0] = cls_loss+lambda_*coord_loss;
  top[0]->mutable_cpu_data()[1] = lambda_*coord_loss;
  DLOG(INFO) << "class loss = "<<cls_loss<<", cls_count = "<<cls_count
             <<", coord loss = "<<coord_loss<<", coord_count = "<<coord_count;
}

template <typename Dtype>
__global__ void DetectionSoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* label, const Dtype* bootstrap_map, Dtype* bottom_diff, const int bottom_dim,
          const int label_dim, const int spatial_dim, const int channels, const bool objectness,
          const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[n * label_dim + s]);
    if (objectness) {
      label_value = min(1,label_value);
    } 
    const int keep_flag = static_cast<int>(bootstrap_map[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * bottom_dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else if ((label_value == 0) && (keep_flag == 0)) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * bottom_dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * bottom_dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void DetectionBoxBackwardGPU(const int nthreads, 
          const Dtype* label, Dtype* bottom_diff, const int bottom_dim,
          const int label_dim, const int spatial_dim, const int coord_num,
          const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * label_dim + s]);
    if ((has_ignore_label_ && label_value == ignore_label_) || (label_value == 0)) {
      for (int c = 0; c < coord_num; ++c) {
        bottom_diff[n * bottom_dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      counts[index] = coord_num;
    }
  }
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* bootstrap_map = bootstrap_map_.gpu_data();
    const Dtype* weight_map = weight_map_.gpu_data();
    const int num = prob_.num();
    const int spatial_dim = prob_.height() * prob_.width();
    const int bottom_dim = bottom[0]->count() / num;
    const int cls_dim = prob_.count() / num;
    const int label_dim = bottom[1]->count() / num;
    const int nthreads = num * spatial_dim;

    for (int i = 0; i < num; i++) {
      caffe_gpu_memcpy(cls_dim * sizeof(Dtype), prob_data+i*cls_dim, bottom_diff+i*bottom_dim);
    }
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* cls_counts = prob_.mutable_gpu_diff();
    DetectionSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, bootstrap_map, bottom_diff, bottom_dim, 
        label_dim, spatial_dim, cls_num_, objectness_, has_ignore_label_, ignore_label_, cls_counts);

    // weihgting the class gradient
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < cls_num_; j++) {
        Dtype* tmp_diff = bottom_diff+i*bottom_dim+j*spatial_dim;
        caffe_gpu_mul<Dtype>(spatial_dim, tmp_diff, weight_map+i*spatial_dim, tmp_diff);
      }
    }
    // gradient of coordinate bottom
    const Dtype* coord_diff_data = coord_diff_.gpu_data();
    const int coord_dim = coord_diff_.count() / num;
    for (int i = 0; i < num; i++) {
      caffe_gpu_memcpy(coord_dim * sizeof(Dtype), coord_diff_data+i*coord_dim, 
                       bottom_diff+i*bottom_dim+cls_dim);
    }
    
    Dtype* coord_counts = coord_diff_.mutable_gpu_diff();
    // the bottom_diff should plus the class feature offset
    DetectionBoxBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, bottom_diff+cls_dim, bottom_dim, label_dim, 
        spatial_dim, coord_num_, has_ignore_label_, ignore_label_, coord_counts);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    Dtype cls_count, coord_count;
    caffe_gpu_asum(nthreads, cls_counts, &cls_count);
    caffe_gpu_asum(nthreads, coord_counts, &coord_count);
    for (int i = 0; i < num; ++i) {
      if (cls_count > 0) {
        caffe_gpu_scal(cls_dim, loss_weight / cls_count, bottom_diff+i*bottom_dim);
      }
      if (coord_count > 0) {
        caffe_gpu_scal(coord_dim, loss_weight*lambda_ / coord_count, 
                       bottom_diff+i*bottom_dim+cls_dim);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionLossLayer);

}  // namespace caffe
