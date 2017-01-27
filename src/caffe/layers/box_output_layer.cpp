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
#include "caffe/layers/box_output_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void BoxOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BoxOutputParameter box_output_param = this->layer_param_.box_output_param();
  fg_thr_ = box_output_param.fg_thr();
  iou_thr_ = box_output_param.iou_thr();
  nms_type_ = box_output_param.nms_type();
  output_proposal_with_score_ = (top.size() == 2);
}

template <typename Dtype>
void BoxOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //dummy reshape
  top[0]->Reshape(1, 5, 1, 1);
  if (output_proposal_with_score_) {
    top[1]->Reshape(1, 6, 1, 1);
  }
}

template <typename Dtype>
vector<vector<Dtype> > nmsMax(const vector<vector<Dtype> >bbs, const float overlap, 
        const bool greedy, const string mode) {
  //bbs[i] = [batch_idx x y w h sc];
  vector<vector<Dtype> > outbbs;
  const int n = bbs.size();
  if (n <= 0) return outbbs;
  // for each i suppress all j st j>i and area-overlap>overlap
  vector<bool> kp(n); 
  for (int i = 0; i < n; i++) kp[i] = true;
  for (int i = 0; i < n; i++){ 
    if(greedy && !kp[i]) continue; 
    for (int j = i+1; j < n; j++) {
      if(kp[j]==0) continue; 
      Dtype o = BoxIOU(bbs[i][1], bbs[i][2], bbs[i][3], bbs[i][4],
                 bbs[j][1], bbs[j][2], bbs[j][3], bbs[j][4], mode); 
      if(o>overlap) kp[j]=false;
    }
  }
  for (int i = 0; i < n; i++) {
    if (kp[i]) {
      outbbs.push_back(bbs[i]);
    }
  }
  return outbbs;
}

template <typename Dtype>
void BoxOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<vector<Dtype> > batch_boxes;
  int num_batch_boxes = 0;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int bottom_num = bottom.size();
  const int cls_num = channels-4;
  float field_whr = this->layer_param_.box_output_param().field_whr();
  float field_xyr = this->layer_param_.box_output_param().field_xyr();
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;
  const int max_nms_num = this->layer_param_.box_output_param().max_nms_num(); 
  const int max_post_nms_num = this->layer_param_.box_output_param().max_post_nms_num(); 
  const float min_size = this->layer_param_.box_output_param().min_size();
  
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().field_h_size());
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().field_w_size());
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().downsample_rate_size());
  vector<float> field_ws, field_hs, downsample_rates;
  for (int i = 0; i < bottom_num; i++) {
    field_ws.push_back(this->layer_param_.box_output_param().field_w(i));
    field_hs.push_back(this->layer_param_.box_output_param().field_h(i));
    downsample_rates.push_back(this->layer_param_.box_output_param().downsample_rate(i));
  }
  
  // bbox mean and std
  bool do_bbox_norm = false;
  vector<float> bbox_means, bbox_stds;
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    do_bbox_norm = true;
    int num_bbox_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_bbox_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_bbox_means,4); CHECK_EQ(num_bbox_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_means.push_back(this->layer_param_.bbox_reg_param().bbox_mean(i));
      bbox_stds.push_back(this->layer_param_.bbox_reg_param().bbox_std(i));
    }
  }
  
  for (int i = 0; i < num; i++) {
    vector<vector<Dtype> > boxes;
    std::vector<std::pair<Dtype, int> > score_idx_vector;
    int bb_count = 0;
    for (int j = 0; j < bottom_num; j++) {
      const Dtype* bottom_data = bottom[j]->cpu_data();
      int bottom_dim = bottom[j]->count() / num;
      int width = bottom[j]->width(), height = bottom[j]->height();
      int img_width = width*downsample_rates[j], img_height = height*downsample_rates[j];
      int spatial_dim = width*height;
      
      for (int id = 0; id < spatial_dim; id++) {
        const int base_idx = i*bottom_dim+id;
        const int coord_idx = base_idx+cls_num*spatial_dim;
        const int h = id / width, w = id % width; 
        Dtype fg_score = -FLT_MAX;
        // get the max score across positive classes
        for (int k = 1; k < cls_num; k++) {
          fg_score = std::max(fg_score,bottom_data[base_idx+k*spatial_dim]);
        }
        fg_score -= bottom_data[base_idx]; // max positive score minus negative score
      
        if (fg_score >= fg_thr_) {
          vector<Dtype> bb(6); 
          Dtype bbx, bby, bbw, bbh;
          bbx = bottom_data[coord_idx];
          bby = bottom_data[coord_idx+spatial_dim];
          bbw = bottom_data[coord_idx+2*spatial_dim];
          bbh = bottom_data[coord_idx+3*spatial_dim];
          
          // bbox de-normalization
          if (do_bbox_norm) {
            bbx *= bbox_stds[0]; bby *= bbox_stds[1];
            bbw *= bbox_stds[2]; bbh *= bbox_stds[3];
            bbx += bbox_means[0]; bby += bbox_means[1];
            bbw += bbox_means[2]; bbh += bbox_means[3];
          }
          
          bbx = std::max(min_xyr,bbx); bbx = std::min(max_xyr,bbx); 
          bby = std::max(min_xyr,bby); bby = std::min(max_xyr,bby);
          bbx = bbx*field_ws[j] + (w+Dtype(0.5))*downsample_rates[j];
          bby = bby*field_hs[j] + (h+Dtype(0.5))*downsample_rates[j];         
        
          bbw = std::max(min_whr,bbw); bbw = std::min(max_whr,bbw); 
          bbh = std::max(min_whr,bbh); bbh = std::min(max_whr,bbh);
          bbw = field_ws[j] * exp(bbw); bbh = field_hs[j] * exp(bbh);
          bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
          bbx = std::max(bbx,Dtype(0)); bby = std::max(bby,Dtype(0));
          bbw = std::min(bbw,img_width-bbx); bbh = std::min(bbh,img_height-bby);
          bb[0] = i; bb[1] = bbx; bb[2] = bby; bb[3] = bbw; bb[4] = bbh; bb[5] = fg_score;
          if (bbw >= min_size && bbh >= min_size) {
            boxes.push_back(bb);
            score_idx_vector.push_back(std::make_pair(fg_score, bb_count++));
          }
        }
      }
    }
    
    DLOG(INFO) << "The number of boxes before NMS: " << boxes.size();
    if (boxes.size()<=0) continue;
    //ranking decreasingly
    std::sort(score_idx_vector.begin(),score_idx_vector.end(),std::greater<std::pair<Dtype, int> >());
    vector<vector<Dtype> > new_boxes;
    for (int kk = 0; kk < boxes.size(); kk++) {
      new_boxes.push_back(boxes[score_idx_vector[kk].second]);
    }
    boxes.clear(); 
    boxes = new_boxes; new_boxes.clear();
    //keep top N boxes before NMS
    if (max_nms_num > 0 && bb_count > max_nms_num) {
      boxes.resize(max_nms_num);
      score_idx_vector.resize(max_nms_num);
    }

    //NMS
    new_boxes = nmsMax(boxes, iou_thr_, true, nms_type_);
    int num_new_boxes =  new_boxes.size();
    if (max_post_nms_num > 0 && num_new_boxes > max_post_nms_num) {
      num_new_boxes = max_post_nms_num;
    }
    for (int kk = 0; kk < num_new_boxes; kk++) {
      batch_boxes.push_back(new_boxes[kk]);
    }
    num_batch_boxes += num_new_boxes;
  }
  
  CHECK_EQ(num_batch_boxes,batch_boxes.size());
  // output rois [batch_idx x1 y1 x2 y2] for roi_pooling layer
  if (num_batch_boxes <= 0) {
    // for special case when there is no box
    top[0]->Reshape(1, 5, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    top_boxes[0]=0; top_boxes[1]=1; top_boxes[2]=1; top_boxes[3]=10; top_boxes[4]=10;
  } else {
    top[0]->Reshape(num_batch_boxes, 5, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    for (int i = 0; i < num_batch_boxes; i++) {
      CHECK_EQ(batch_boxes[i].size(),6);
      top_boxes[i*5] = batch_boxes[i][0];
      top_boxes[i*5+1] = batch_boxes[i][1];
      top_boxes[i*5+2] = batch_boxes[i][2];
      top_boxes[i*5+3] = batch_boxes[i][1]+batch_boxes[i][3];
      top_boxes[i*5+4] = batch_boxes[i][2]+batch_boxes[i][4];
    }
  }
  // output proposals+scores [batch_idx x1 y1 x2 y2 score] for proposal detection
  if (output_proposal_with_score_) {
    if (num_batch_boxes <= 0) {
      // for special case when there is no box
      top[1]->Reshape(1, 6, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      caffe_set(top[1]->count(), Dtype(0), top_boxes_scores); 
    } else {
      const int top_dim = 6;
      top[1]->Reshape(num_batch_boxes, top_dim, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      for (int i = 0; i < num_batch_boxes; i++) {
        CHECK_EQ(batch_boxes[i].size(),6);
        top_boxes_scores[i*top_dim] = batch_boxes[i][0];
        top_boxes_scores[i*top_dim+1] = batch_boxes[i][1];
        top_boxes_scores[i*top_dim+2] = batch_boxes[i][2];
        top_boxes_scores[i*top_dim+3] = batch_boxes[i][1]+batch_boxes[i][3];
        top_boxes_scores[i*top_dim+4] = batch_boxes[i][2]+batch_boxes[i][4];
        top_boxes_scores[i*top_dim+5] = batch_boxes[i][5];
      }
    }
  }
}

INSTANTIATE_CLASS(BoxOutputLayer);
REGISTER_LAYER_CLASS(BoxOutput);

}  // namespace caffe
