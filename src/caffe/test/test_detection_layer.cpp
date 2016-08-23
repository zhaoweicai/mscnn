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
    
//typedef ::testing::Types<FloatGPU, DoubleGPU> TestDtypesGPU;
//typedef ::testing::Types<DoubleGPU> TestDtypesGPU;
typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;

template <typename TypeParam>
class DetectionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DetectionLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 9, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(4, 6, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int num = blob_bottom_label_->num();
    int height = blob_bottom_label_->height(), width = blob_bottom_label_->width();
    int label_dim = blob_bottom_label_->count() / num;
    int spatial_dim = height*width;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        blob_bottom_label_->mutable_cpu_data()[i*label_dim+j] = caffe_rng_rand() % 5;
      }
      for (int j = spatial_dim; j < label_dim-spatial_dim; j++) {
        blob_bottom_label_->mutable_cpu_data()[i*label_dim+j] = caffe_rng_rand() % 64;
      }
      for (int j = label_dim-spatial_dim; j < label_dim; j++) {
        blob_bottom_label_->mutable_cpu_data()[i*label_dim+j] = (caffe_rng_rand()%100)/Dtype(100);
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~DetectionLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(DetectionLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(DetectionLayerTest, TestDtypesGPU);

TYPED_TEST(DetectionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  DetectionParameter* detection_param =
      layer_param.mutable_detection_param();
  detection_param->set_cls_num(5);
  detection_param->set_coord_num(4);
  detection_param->set_lambda(1.2);
  detection_param->set_field_h(64);
  detection_param->set_field_w(32);
  detection_param->set_field_r(2);
  detection_param->set_downsample_rate(8);
  detection_param->set_bb_smooth(false);
  detection_param->set_bg_threshold(0.2);
  detection_param->set_bg_multiple(4);
  detection_param->set_sample_mode("mixture");
  detection_param->set_objectness(false);
  detection_param->set_iou_weighted(false);
  detection_param->set_pos_neg_weighted(true);

  DetectionLayer<Dtype> layer(layer_param);
  
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

/*TYPED_TEST(DetectionLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionParameter* detection_param =
      layer_param.mutable_detection_param();
  detection_param->set_cls_num(5);
  detection_param->set_coord_num(4);
  detection_param->set_lambda(1.2);
  detection_param->set_field_h(64);
  detection_param->set_field_w(32);
  detection_param->set_downsample_rate(32);
  detection_param->set_bb_smooth(true);
  detection_param->set_bg_threshold(0.5);
  detection_param->set_bg_multiple(4);
  layer_param.mutable_loss_param()->set_normalize(true);
  // First, compute the loss with all labels
  scoped_ptr<DetectionLayer<Dtype> > layer(
      new DetectionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {1, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new DetectionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(5 * full_loss, accum_loss, 1e-4);
}

TYPED_TEST(DetectionLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionParameter* detection_param =
      layer_param.mutable_detection_param();
  detection_param->set_cls_num(5);
  detection_param->set_coord_num(4);
  detection_param->set_lambda(1.2);
  detection_param->set_field_h(64);
  detection_param->set_field_w(32);
  detection_param->set_bb_smooth(true);
  detection_param->set_downsample_rate(32);
  detection_param->set_bg_threshold(0.5);
  detection_param->set_bg_multiple(4);
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(1);
  DetectionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(DetectionLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionParameter* detection_param =
      layer_param.mutable_detection_param();
  detection_param->set_cls_num(5);
  detection_param->set_coord_num(4);
  detection_param->set_lambda(1.2);
  detection_param->set_field_h(64);
  detection_param->set_field_w(32);
  detection_param->set_downsample_rate(32);
  detection_param->set_bb_smooth(true);
  detection_param->set_bg_threshold(0.5);
  detection_param->set_bg_multiple(4);
  layer_param.mutable_loss_param()->set_normalize(false);
  DetectionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}*/

}  // namespace caffe
