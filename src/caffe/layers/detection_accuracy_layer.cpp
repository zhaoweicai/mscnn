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
#include "caffe/layers/detection_accuracy_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void DetectionAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DetectionAccuracyParameter detect_acc_param = this->layer_param_.detection_accuracy_param();
  cls_num_ = detect_acc_param.cls_num();
  coord_num_ = detect_acc_param.coord_num();
  field_h_ = detect_acc_param.field_h();
  field_w_ = detect_acc_param.field_w();
  downsample_rate_ = detect_acc_param.downsample_rate();
  top_k_ = detect_acc_param.top_k();
  objectness_ = detect_acc_param.objectness();

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void DetectionAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num(); int channels = bottom[0]->channels();
  int height = bottom[0]->height(); int width = bottom[0]->width();
  CHECK_EQ(num,bottom[1]->num()); 
  CHECK_EQ(height,bottom[1]->height()); CHECK_EQ(width,bottom[1]->width());
  CHECK((cls_num_+coord_num_)==channels) << "the channels dimensions don't match" << std::endl;
  if (objectness_) CHECK_EQ(2,cls_num_);
  top[0]->Reshape(1, 1, 1, 2);
  if (top.size() >= 2) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void DetectionAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int height = bottom[0]->height(); int width = bottom[0]->width();
  int spatial_dim = height*width;
  int bottom_dim = bottom[0]->count() / num;
  int coord_dim = spatial_dim*coord_num_;
  int cls_dim = spatial_dim*cls_num_;
  int label_dim = bottom[1]->count() / num;
  CHECK((cls_dim+coord_dim)==bottom_dim) << "the bottom dimensions don't fit" << std::endl;
  CHECK((2*spatial_dim+coord_dim)==label_dim) << "the label dimensions don't fit" << std::endl;
  DetectionAccuracyParameter detect_acc_param = this->layer_param_.detection_accuracy_param();
  float field_whr = detect_acc_param.field_whr();
  float field_xyr = detect_acc_param.field_xyr();
  float bg_threshold = detect_acc_param.bg_threshold();
  
  // The accuracy forward pass 
  Dtype accuracy = 0, fore_accuracy = 0;
  int acc_count = 0, fore_count = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int label_value = static_cast<int>(label[i * label_dim + j]);
      if (objectness_) {
        label_value = std::min(1,label_value);
      }
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (label_value == 0 && label[i*label_dim+j+spatial_dim+coord_dim] >= bg_threshold) {
        continue; // ignored bounding boxes
      }
      if (label_value != 0) {
        ++fore_count;
      }
      ++acc_count;
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < cls_num_; k++) {
        bottom_data_vector.push_back(
            std::make_pair(bottom_data[i * bottom_dim + k * spatial_dim + j], k));
      }
      std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (label_value != 0) {
            ++fore_accuracy; 
          }
          break;
        }
      }
    }
  }
  
  if (acc_count != 0) {
    accuracy /= acc_count;
  } else {
    accuracy = Dtype(-1);
  }
  if (fore_count != 0) {
    fore_accuracy /= fore_count;
  } else {
    fore_accuracy = Dtype(-1);
  }

  int coord_count = 0;
  Dtype coord_iou = 0;
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int label_index = i*label_dim + h*width + w;
        const int label_value = static_cast<int>(label[label_index]);
        if ((has_ignore_label_ && label_value == ignore_label_) || (label_value==0))  {
          continue;
        }
        Dtype tx, ty, tw, th; 
        int coord_idx = i*bottom_dim + cls_dim + h*width + w;
        tx = bottom_data[coord_idx];
        ty = bottom_data[coord_idx+spatial_dim];
        tw = bottom_data[coord_idx+2*spatial_dim];
        th = bottom_data[coord_idx+3*spatial_dim];
                
        tx = std::max(min_xyr,tx); tx = std::min(max_xyr,tx); 
        ty = std::max(min_xyr,ty); ty = std::min(max_xyr,ty);
        tx = tx*field_w_ + (w+Dtype(0.5))*downsample_rate_;
        ty = ty*field_h_ + (h+Dtype(0.5))*downsample_rate_;
        
        tw = std::max(min_whr,tw); tw = std::min(max_whr,tw); 
        th = std::max(min_whr,th); th = std::min(max_whr,th);
        tw = field_w_ * exp(tw); th = field_h_ * exp(th);
        tx = tx - tw/Dtype(2); ty = ty - th/Dtype(2);

        Dtype gx, gy, gw, gh;
        gx = label[label_index+spatial_dim]; gy = label[label_index+2*spatial_dim];
        gw = label[label_index+3*spatial_dim]; gh = label[label_index+4*spatial_dim];
        gx = gx - gw/Dtype(2); gy = gy - gh/Dtype(2);
        
        Dtype iou = BoxIOU(tx,ty,tw,th,gx,gy,gw,gh,"IOU");
        coord_iou += iou;
        coord_count++; 
      }
    }
  }
  
  if (coord_count != 0) { 
    coord_iou /= coord_count;
  } else {
    coord_iou = Dtype(-1);
  }

  DLOG(INFO) << "Acc = "<<accuracy<<", ForeAcc = "<<fore_accuracy<<", IOU = "<<coord_iou;
  top[0]->mutable_cpu_data()[0] = accuracy;
  top[0]->mutable_cpu_data()[1] = fore_accuracy;
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = coord_iou;
  }
}

INSTANTIATE_CLASS(DetectionAccuracyLayer);
REGISTER_LAYER_CLASS(DetectionAccuracy);

}  // namespace caffe
