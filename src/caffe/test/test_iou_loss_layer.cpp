// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;
typedef ::testing::Types<CPUDevice<float>, CPUDevice<double> > TestDtypesCPU;

template <typename TypeParam>
class IOULossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IOULossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 12, 1, 1)),
        blob_pesudo_mapping_(new Blob<Dtype>(10, 4, 1, 1)),
        blob_bottom_anchors_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_gt_boxes_(new Blob<Dtype>(10, 6, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // check
    const int batch_num = 4;
    const int cls_num = 3;
    const int anchor_dim = 5;
    const int gt_dim = 6;
    const int coord_dim = 4;
    const Dtype min_size = 32;
    const int num = blob_bottom_data_->num();
    CHECK_EQ(num,blob_pesudo_mapping_->num());
    CHECK_EQ(num,blob_bottom_anchors_->num());
    CHECK_EQ(num,blob_bottom_gt_boxes_->num());
    CHECK_EQ(blob_bottom_data_->channels(),cls_num*coord_dim);
    CHECK_EQ(blob_pesudo_mapping_->channels(),coord_dim);
    CHECK_EQ(anchor_dim,blob_bottom_anchors_->channels());
    CHECK_EQ(gt_dim,blob_bottom_gt_boxes_->channels());
    // fill the values
    /*FillerParameter const_filler_param;
    const_filler_param.set_value(-1.);
    ConstantFiller<Dtype> const_filler(const_filler_param);*/
    FillerParameter filler_param;
    filler_param.set_std(0.5);
    GaussianFiller<Dtype> filler(filler_param);
    // prediction
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    
    // pseudo-mapping
    filler_param.set_std(0.1);
    GaussianFiller<Dtype> filler2(filler_param);
    filler2.Fill(this->blob_pesudo_mapping_);
    
    // anchors: (batch_id, x1, y1, x2, y2)
    for (int i = 0; i < num; ++i) {
      blob_bottom_anchors_->mutable_cpu_data()[i*anchor_dim] = caffe_rng_rand() % batch_num;
      Dtype x, y, w, h;
      x = caffe_rng_rand() % 200; y = caffe_rng_rand() % 200;
      w = caffe_rng_rand() % 300 + min_size; h = caffe_rng_rand() % 300 + min_size;
      blob_bottom_anchors_->mutable_cpu_data()[i*anchor_dim+1] = x;
      blob_bottom_anchors_->mutable_cpu_data()[i*anchor_dim+2] = y;
      blob_bottom_anchors_->mutable_cpu_data()[i*anchor_dim+3] = x+w;
      blob_bottom_anchors_->mutable_cpu_data()[i*anchor_dim+4] = y+h;
    }
    blob_bottom_vec_.push_back(blob_bottom_anchors_);
    
    // gt: (label, x1, y1, x2, y2, overlap), map from gt by pseudo-mapping
    for (int i = 0; i < num; ++i) {
      Dtype label = caffe_rng_rand() % cls_num;
      //Dtype label = 1;
      // gt
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim] = label;
      Dtype x, y, w, h, anchor_w, anchor_h;
      const Dtype* anchor_data = blob_bottom_anchors_->cpu_data();
      const Dtype* map_data = blob_pesudo_mapping_->cpu_data();
      anchor_w = anchor_data[i*anchor_dim+3]-anchor_data[i*anchor_dim+1]+1;
      anchor_h = anchor_data[i*anchor_dim+4]-anchor_data[i*anchor_dim+2]+1;
      x = map_data[i*coord_dim]*anchor_w+anchor_data[i*anchor_dim+1];
      y = map_data[i*coord_dim+1]*anchor_h+anchor_data[i*anchor_dim+2];
      w = std::max(min_size,Dtype(exp(map_data[i*coord_dim+2])*anchor_w));
      h = std::max(min_size,Dtype(exp(map_data[i*coord_dim+3])*anchor_h));
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim+1] = x;
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim+2] = y;
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim+3] = x+w-1;
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim+4] = y+h-1;
      blob_bottom_gt_boxes_->mutable_cpu_data()[i*gt_dim+5] = (caffe_rng_rand()%100)/Dtype(100);
    }
    blob_bottom_vec_.push_back(blob_bottom_gt_boxes_);
    
    // print out the data
    const int pred_dim = cls_num*coord_dim;
    std::cout<<"========================== print out data for checking ======================="<<std::endl;
    for (int i = 0; i < num; i++) {
      std::cout<<"---------- id: "<<i<<" ----------"<<std::endl;
      std::cout<<"pred_data: ";
      for (int j = 0; j < pred_dim; j++) 
        std::cout<<blob_bottom_data_->cpu_data()[i*pred_dim+j]<<", ";
      cout<<std::endl;
      std::cout<<"anchor: ";
      for (int j = 0; j < anchor_dim; j++) 
        std::cout<<blob_bottom_anchors_->cpu_data()[i*anchor_dim+j]<<", ";
      cout<<std::endl;
      std::cout<<"gt_boxes: ";
      for (int j = 0; j < gt_dim; j++) 
        std::cout<<blob_bottom_gt_boxes_->cpu_data()[i*gt_dim+j]<<", ";
      cout<<std::endl;
    }


    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~IOULossLayerTest() {
    delete blob_bottom_data_;
    delete blob_pesudo_mapping_;
    delete blob_bottom_anchors_;
    delete blob_bottom_gt_boxes_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_pesudo_mapping_;
  Blob<Dtype>* const blob_bottom_anchors_;
  Blob<Dtype>* const blob_bottom_gt_boxes_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IOULossLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(IOULossLayerTest, TestDtypesGPU);

TYPED_TEST(IOULossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  const Dtype kLossWeight = 3;
  layer_param.add_loss_weight(kLossWeight);
  IOULossParameter* iou_loss_param =
      layer_param.mutable_iou_loss_param();
  iou_loss_param->add_bbox_mean(0);
  iou_loss_param->add_bbox_mean(0.1);
  iou_loss_param->add_bbox_mean(0.2);
  iou_loss_param->add_bbox_mean(0);
  iou_loss_param->add_bbox_std(0.1);
  iou_loss_param->add_bbox_std(0.4);
  iou_loss_param->add_bbox_std(0.3);
  iou_loss_param->add_bbox_std(0.2);
  IOULossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-4, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
