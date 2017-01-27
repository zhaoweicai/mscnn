// ------------------------------------------------------------------
// MS-CNN
// Copyright (c) 2016 The Regents of the University of California
// see mscnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_IMAGE_GT_DATA_LAYER_HPP_
#define CAFFE_IMAGE_GT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ImageGtDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageGtDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageGtDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageGtData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 2; }

  void BoundingboxAffine(vector<vector<float> >& bbs,
        float w_scale, float h_scale, float w_shift, float h_shift);

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void ShuffleList();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum ImageGtField { X1, Y1, X2, Y2, LABEL, IGNORE, NUM };
  vector<vector<vector<float> > > windows_;
  vector<vector<vector<float> > > roni_windows_;
  vector<Dtype> mean_values_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
  vector<int> downsample_rates_;
  vector<int> field_ws_;
  vector<int> field_hs_;
  int label_channel_;
  int label_blob_num_;
  vector<int> image_list_;
  int list_id_;
  bool output_gt_boxes_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_GT_DATA_LAYER_HPP_
