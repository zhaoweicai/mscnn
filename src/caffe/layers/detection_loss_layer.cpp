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
void DetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DetectionLossParameter detect_param = this->layer_param_.detection_loss_param();
  cls_num_ = detect_param.cls_num();
  coord_num_ = detect_param.coord_num();
  lambda_ = detect_param.lambda();
  field_h_ = detect_param.field_h();
  field_w_ = detect_param.field_w();
  field_whr_ = detect_param.field_whr();
  field_xyr_ = detect_param.field_xyr();
  downsample_rate_ = detect_param.downsample_rate();
  bb_smooth_ = detect_param.bb_smooth();
  bg_threshold_ = detect_param.bg_threshold();
  bg_multiple_ = detect_param.bg_multiple();
  sample_mode_ = detect_param.sample_mode();
  objectness_ = detect_param.objectness();
  iou_weighted_ = detect_param.iou_weighted();
  pos_neg_weighted_ = detect_param.pos_neg_weighted();

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  
  // split the bottom into two blobs (classes and coordinates)
  int num = bottom[0]->num(); int channels = bottom[0]->channels();
  int height = bottom[0]->height(); int width = bottom[0]->width();
  CHECK((cls_num_+coord_num_)==channels) << "the channels dimensions don't match" << std::endl;
  
  cls_bottom_.Reshape(num, cls_num_, height, width);
  DLOG(INFO) << "cls_bottom size: " << cls_bottom_.num() << ","
      << cls_bottom_.channels() << "," << cls_bottom_.height() << ","
      << cls_bottom_.width();
  
  coord_bottom_.Reshape(num, coord_num_, height, width);
  coord_diff_.Reshape(num, coord_num_, height, width);
  DLOG(INFO) << "coord_bottom size: " << coord_bottom_.num() << ","
      << coord_bottom_.channels() << "," << coord_bottom_.height() << ","
      << coord_bottom_.width();
  
  bootstrap_map_.Reshape(num, 1, height, width);
  weight_map_.Reshape(num, 1, height, width);
  
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&cls_bottom_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int height = bottom[0]->height(); int width = bottom[0]->width();
  CHECK_EQ(num,bottom[1]->num()); 
  CHECK_EQ(height,bottom[1]->height()); CHECK_EQ(width,bottom[1]->width());
  LossLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(1,1,1,2);

  cls_bottom_.Reshape(num, cls_num_, height, width);
  CHECK_EQ(num,softmax_bottom_vec_[0]->num()); 
  CHECK_EQ(height,softmax_bottom_vec_[0]->height()); 
  CHECK_EQ(width,softmax_bottom_vec_[0]->width());
  if (objectness_) CHECK_EQ(2,cls_num_);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  
  coord_bottom_.Reshape(num, coord_num_, height, width);
  coord_diff_.Reshape(num, coord_num_, height, width);
  bootstrap_map_.Reshape(num, 1, height, width);
  weight_map_.Reshape(num, 1, height, width);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // copy the bottom into two blobs (classes and coordinates)
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num(); int channels = bottom[0]->channels();
  int height = bottom[0]->height(); int width = bottom[0]->width();
  int spatial_dim = height*width;
  int bottom_dim = bottom[0]->count() / num;
  int coord_dim = coord_bottom_.count() / num;
  int cls_dim = cls_bottom_.count() / num;
  int label_dim = bottom[1]->count() / num;
  CHECK((cls_num_+coord_num_)==channels) << "the channels dimensions don't fit" << std::endl;
   
  CHECK_EQ(cls_bottom_.count(),num*cls_num_*spatial_dim); 
  Dtype* cls_bottom_data = cls_bottom_.mutable_cpu_data();
  Dtype* coord_bottom_data = coord_bottom_.mutable_cpu_data();
  for (int i = 0; i < num; i++) {
    caffe_copy(cls_dim, bottom_data+i*bottom_dim, cls_bottom_data+i*cls_dim);
    caffe_copy(coord_dim, bottom_data+i*bottom_dim+cls_dim, coord_bottom_data+i*coord_dim);
  }
    
  //build the bootstrap map
  Dtype* bootstrap_map_data = bootstrap_map_.mutable_cpu_data();
  caffe_set(bootstrap_map_.count(), Dtype(0), bootstrap_map_data);
  int total_neg_num = 0, total_pos_num = 0, positive_num, keep_num = 0;
  bool sample_all_flag = true;
  
  for (int i = 0; i < num; ++i) {
    vector<int> instance_nums(cls_num_);
    positive_num = 0;
    for (int c = 0; c < cls_num_; c++) instance_nums[c] = 0;
    for (int j = 0; j < spatial_dim; j++) {
      int base_index = i*label_dim+j;
      int label_value = static_cast<int>(label[base_index]);
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
        const int label_value = static_cast<int>(label[base_index]);
        if ((label_value==0) && (label[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
          if (bootstrap_map_data[i*spatial_dim+idx] == 0) {keep_num++;}
          bootstrap_map_data[i*spatial_dim+idx] = 1;
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
          const int label_value = static_cast<int>(label[base_index]);
          if ((label_value==0) && (label[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
            loc_vector.push_back(std::make_pair(bottom_data[i*bottom_dim+j+k*spatial_dim], j));
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
          if (bootstrap_map_data[i*spatial_dim+idx] == 0) {keep_num++;}
          bootstrap_map_data[i*spatial_dim+idx] = 1;
        }
      }
    } 
    if (sample_all_flag) {
      for (int j = 0; j < spatial_dim; j++) {
        int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label[base_index]);
        if ((label_value==0) && (label[base_index+spatial_dim+coord_dim] < bg_threshold_)) {
          if (bootstrap_map_data[i*spatial_dim+j] == 0) {keep_num++;}
          bootstrap_map_data[i*spatial_dim+j] = 1;
        }
      }
    }
    total_neg_num += instance_nums[0]; total_pos_num += positive_num;
  }
  int keep_num_check = caffe_cpu_asum(num*spatial_dim,bootstrap_map_data);
  CHECK_EQ(keep_num, keep_num_check);
  DLOG(INFO)<<"cls num = "<<cls_num_<<", total positive = "<<total_pos_num
            <<", total negative = "<<total_neg_num<<", keep = "<<keep_num;
    
  //setup weight map
  Dtype* weight_map_data = weight_map_.mutable_cpu_data();
  caffe_set(weight_map_.count(), Dtype(1), weight_map_data);
  if (iou_weighted_) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label[base_index]);
        if (label_value != 0) {
          weight_map_data[i*spatial_dim+j] = label[base_index+spatial_dim+coord_dim];
        }
      }
    }
  }
  
  Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
  if (pos_neg_weighted_) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label[base_index]);
        if (label_value != 0) {
          pos_weight_sum += weight_map_data[i*spatial_dim+j]; 
        } else if (bootstrap_map_data[i*spatial_dim+j] == 1) {
          neg_weight_sum += weight_map_data[i*spatial_dim+j]; 
        }
      }
    }
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = i*label_dim+j;
        const int label_value = static_cast<int>(label[base_index]);
        Dtype fg_weight = Dtype(1)/(1+bg_multiple_);
        int sample_num = keep_num+total_pos_num;
        if (label_value != 0) {
          if (pos_weight_sum != 0) {
            weight_map_data[i*spatial_dim+j] *= (fg_weight*sample_num/pos_weight_sum); 
          }
        } else {
          if (pos_weight_sum != 0) {
            weight_map_data[i*spatial_dim+j] *= ((1-fg_weight)*sample_num/neg_weight_sum); 
          }
        }
      }
    }
  }
    
  // The forward pass computes the softmax prob values.
  CHECK_EQ(num,softmax_bottom_vec_[0]->num()); 
  CHECK_EQ(height,softmax_bottom_vec_[0]->height()); CHECK_EQ(width,softmax_bottom_vec_[0]->width());
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  
  const Dtype* prob_data = prob_.cpu_data();
  CHECK_EQ(num,prob_.num()); CHECK_EQ(spatial_dim, prob_.height() * prob_.width());
  
  int cls_count = 0;
  Dtype cls_loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int base_index = i*spatial_dim+j;
      int label_value = static_cast<int>(label[i * label_dim + j]);
      if (objectness_) {
        label_value = std::min(1,label_value);
      } 
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      const int keep_flag = static_cast<int>(bootstrap_map_data[base_index]);
      if (label_value != 0) {CHECK_EQ(keep_flag,0);}
      if ((label_value == 0) && (keep_flag == 0)) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.channels());
      cls_loss -= log(std::max(prob_data[i * cls_dim + label_value * spatial_dim + j],
                           Dtype(FLT_MIN))) * weight_map_data[base_index];
      ++cls_count;
    }
  }
  
  // the forward pass computes euclidean loss 
  int coord_count = 0;
  Dtype coord_loss = 0;
  int label_height = bottom[1]->height(), label_width = bottom[1]->width();
  Dtype* coord_diff_data = coord_diff_.mutable_cpu_data();
  caffe_set(coord_diff_.count(), Dtype(0), coord_diff_data);
  Dtype min_whr = log(Dtype(1)/field_whr_), max_whr = log(Dtype(field_whr_));
  Dtype min_xyr = Dtype(-1)/field_xyr_, max_xyr = Dtype(1)/field_xyr_;

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < label_height; ++h) {
      for (int w = 0; w < label_width; ++w) {
        int label_index = i*label_dim + h*label_width + w;
        const int label_value = static_cast<int>(label[label_index]);
        if ((has_ignore_label_ && label_value == ignore_label_) || (label_value==0))  {
          continue;
        }
        Dtype gx, gy, gw, gh; 
        gx = (label[label_index+spatial_dim]-(w+Dtype(0.5))*downsample_rate_) / field_w_;
        gy = (label[label_index+2*spatial_dim]-(h+Dtype(0.5))*downsample_rate_) / field_h_;
        gw = log(std::max(label[label_index+3*spatial_dim],Dtype(2)) / field_w_);
        gh = log(std::max(label[label_index+4*spatial_dim],Dtype(2)) / field_h_);
       
        // euclidean gradient
        Dtype tx, ty, tw, th;
        int coord_idx = i*coord_dim + h*label_width + w;
        tx = coord_bottom_data[coord_idx];
        ty = coord_bottom_data[coord_idx+spatial_dim];
        tw = coord_bottom_data[coord_idx+2*spatial_dim];
        th = coord_bottom_data[coord_idx+3*spatial_dim];
        tx = std::max(min_xyr,tx); tx = std::min(max_xyr,tx); 
        ty = std::max(min_xyr,ty); ty = std::min(max_xyr,ty); 
        tw = std::max(min_whr,tw); tw = std::min(max_whr,tw); 
        th = std::max(min_whr,th); th = std::min(max_whr,th); 

        coord_diff_data[coord_idx] = tx-gx; 
        coord_diff_data[coord_idx+spatial_dim] = ty-gy;
        coord_diff_data[coord_idx+2*spatial_dim] = tw-gw;
        coord_diff_data[coord_idx+3*spatial_dim] = th-gh;
        coord_count += coord_num_; 
      }
    }
  }

  if (bb_smooth_) {
    for (int i = 0; i < coord_diff_.count(); ++i) {
      if (coord_diff_data[i] <= -1) { 
        coord_loss += (std::abs(coord_diff_data[i])-Dtype(0.5));
        coord_diff_data[i] = Dtype(-1);
      } else if (coord_diff_data[i] < 1) {
        coord_loss += (coord_diff_data[i]*coord_diff_data[i]/Dtype(2));
      } else {
        coord_loss += (std::abs(coord_diff_data[i])-Dtype(0.5));
        coord_diff_data[i] = Dtype(1);
      }
    }
  } else {
    coord_loss = caffe_cpu_dot(coord_diff_.count(), coord_diff_.cpu_data(), coord_diff_.cpu_data());
    coord_loss /= Dtype(2);
  }

  // normalize
  if (cls_count == 0) cls_loss = Dtype(0);
  else cls_loss /= cls_count;
  if (coord_count == 0) coord_loss = Dtype(0);
  else coord_loss /= coord_count;
  
  // combine the loss
  top[0]->mutable_cpu_data()[0] = cls_loss+lambda_*coord_loss;
  top[0]->mutable_cpu_data()[1] = lambda_*coord_loss;
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* bootstrap_map_data = bootstrap_map_.cpu_data();
    const Dtype* weight_map_data = weight_map_.cpu_data();
    int num = prob_.num();
    int bottom_dim = bottom[0]->count() / num;
    int cls_dim = prob_.count() / num;
    int label_dim = bottom[1]->count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int cls_count = 0, coord_count = 0;
    // gradient of class bottom
    for (int i = 0; i < num; i++) {
      caffe_copy(cls_dim, prob_data+i*cls_dim, bottom_diff+i*bottom_dim);
    }
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        int label_value = static_cast<int>(label[i * label_dim + j]);
        if (objectness_) {
          label_value = std::min(1,label_value);
        } 
        const int keep_flag = static_cast<int>(bootstrap_map_data[i*spatial_dim+j]);
        DCHECK_LT(label_value, prob_.channels());
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < cls_num_; ++c) {
            bottom_diff[i * bottom_dim + c * spatial_dim + j] = 0;
          }
        } else if ((label_value == 0) && (keep_flag == 0)) {
          for (int c = 0; c < cls_num_; ++c) {
            bottom_diff[i * bottom_dim + c * spatial_dim + j] = 0;
          }
        } else {
          bottom_diff[i * bottom_dim + label_value * spatial_dim + j] -= 1;
          ++cls_count;
        }
      }
    }
    
    // weighting the class gradient
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < cls_num_; j++) {
        Dtype* tmp_diff = bottom_diff+i*bottom_dim+j*spatial_dim;
        caffe_mul(spatial_dim, tmp_diff, weight_map_data+i*spatial_dim, tmp_diff);
      }
    }
    
    //gradient of coordinate bottom
    const Dtype* coord_diff_data = coord_diff_.cpu_data();
    int coord_dim = coord_diff_.count() / num;
    CHECK_EQ(bottom_dim,cls_dim+coord_dim);
    for (int i = 0; i < num; i++) {
      caffe_copy(coord_dim, coord_diff_data+i*coord_dim,
           bottom_diff+i*bottom_dim+cls_dim);
    }
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i*label_dim+j]);
        if ((has_ignore_label_ && label_value == ignore_label_) || (label_value==0)) {
          for (int c = 0; c < coord_num_; ++c) {
            bottom_diff[i * bottom_dim + (c+cls_num_) * spatial_dim + j] = 0;
          }
          continue;
        }
        coord_count += coord_num_;
      }
    }
    
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    for (int i = 0; i < num; ++i) {
      if (cls_count > 0) {
        caffe_scal(cls_dim, loss_weight / cls_count, bottom_diff+i*bottom_dim);
      }
      if (coord_count > 0) {
        caffe_scal(coord_dim, loss_weight*lambda_ / coord_count, bottom_diff+i*bottom_dim+cls_dim);
      }
    } 
  }
}

#ifdef CPU_ONLY
STUB_GPU(DetectionLossLayer);
#endif

INSTANTIATE_CLASS(DetectionLossLayer);
REGISTER_LAYER_CLASS(DetectionLoss);

}  // namespace caffe
