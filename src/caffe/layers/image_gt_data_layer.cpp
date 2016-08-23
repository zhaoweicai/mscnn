// ------------------------------------------------------------------
// MS-CNN
// Copyright (c) 2016 The Regents of the University of California
// see mscnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > ImageGtDataParameter

namespace caffe {

template <typename Dtype>
ImageGtDataLayer<Dtype>::~ImageGtDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageGtDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_gts
  //    label ignore x1 y1 x2 y2
  //    num_roni
  //    x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  batch size: "
      << this->layer_param_.image_gt_data_param().batch_size() << std::endl
      << "  cache_images: "
      << this->layer_param_.image_gt_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.image_gt_data_param().root_folder();

  cache_images_ = this->layer_param_.image_gt_data_param().cache_images();
  string root_folder = this->layer_param_.image_gt_data_param().root_folder();

  // reset the random generator seed
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  const int fg_img_multiple = this->layer_param_.image_gt_data_param().fg_img_multiple();
  int list_index = 0;

  std::ifstream infile(this->layer_param_.image_gt_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.image_gt_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels, img_height, img_width, template_height, template_width;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0]; img_height = image_size[1]; img_width = image_size[2];

    bool fg_img_flag = false;
    // read each box
    int num_windows;
    vector<vector<float> > windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, ignore, x1, y1, x2, y2;
      infile >> label >> ignore >> x1 >> y1 >> x2 >> y2;

      vector<float> window(ImageGtDataLayer::NUM);
      window[ImageGtDataLayer::LABEL] = label;
      window[ImageGtDataLayer::IGNORE] = ignore;
      window[ImageGtDataLayer::X1] = x1;
      window[ImageGtDataLayer::Y1] = y1;
      window[ImageGtDataLayer::X2] = x2;
      window[ImageGtDataLayer::Y2] = y2;
      
      //check if fg image
      if (ignore == 0) {
        fg_img_flag = true;
      }

      // add window to list
      label = window[ImageGtDataLayer::LABEL];
      CHECK_GT(label, 0);
      windows.push_back(window);
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }

    // region of non-interest windows
    int num_roni_windows;
    vector<vector<float> > roni_windows;
    infile >> num_roni_windows;
    for (int i = 0; i < num_roni_windows; ++i) {
      int x1, y1, x2, y2;      
      infile >> x1 >> y1 >> x2 >> y2;

      vector<float> roni_window(ImageGtDataLayer::NUM-2);
      roni_window[ImageGtDataLayer::X1] = x1;
      roni_window[ImageGtDataLayer::Y1] = y1;
      roni_window[ImageGtDataLayer::X2] = x2;
      roni_window[ImageGtDataLayer::Y2] = y2;
      roni_windows.push_back(roni_window);
    }
    
    int multiple = fg_img_flag ? fg_img_multiple:1;
    for (int i = 0; i < multiple; i++) {
      image_database_.push_back(std::make_pair(image_path, image_size));
      image_list_.push_back(list_index++);
      if (cache_images_) {
        Datum datum;
        if (!ReadFileToDatum(image_path, &datum)) {
          LOG(ERROR) << "Could not open or find file " << image_path;
          return;
        }
        image_database_cache_.push_back(std::make_pair(image_path, datum));
      }
      windows_.push_back(windows);
      roni_windows_.push_back(roni_windows);
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows
          << ", RONI windows: "<< num_roni_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;
  LOG(INFO) << "Number of colleting images: " << list_index;
  CHECK_EQ(windows_.size(),image_list_.size());
  CHECK_EQ(windows_.size(),image_database_.size());
  CHECK_EQ(windows_.size(),roni_windows_.size());
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }
  
  if (this->layer_param_.image_gt_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleList();
  }
  list_id_ = 0;
  
  template_width = img_width; template_height = img_height;
  if (this->layer_param_.image_gt_data_param().has_resize_width()
      && this->layer_param_.image_gt_data_param().has_resize_height()) {
    template_width = this->layer_param_.image_gt_data_param().resize_width();
    template_height = this->layer_param_.image_gt_data_param().resize_height();
  }
  if (this->layer_param_.image_gt_data_param().has_crop_width()
      && this->layer_param_.image_gt_data_param().has_crop_height()) {
    int crop_width = this->layer_param_.image_gt_data_param().crop_width();
    int crop_height = this->layer_param_.image_gt_data_param().crop_height();
    if (crop_width > 0) {
      CHECK_LE(crop_width, template_width); 
      template_width = crop_width;
    }
    if (crop_height > 0) {
      CHECK_LE(crop_height, template_height);
      template_height = crop_height;
    }
  }
  // image reshape
  const int batch_size = this->layer_param_.image_gt_data_param().batch_size();
  top[0]->Reshape(batch_size, channels, template_height, template_width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(batch_size, channels, template_height, template_width);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label reshape (on channel axis, [label x y w h iou])
  label_channel_ = 2+this->layer_param_.image_gt_data_param().coord_num();
  label_blob_num_ = this->layer_param_.image_gt_data_param().downsample_rate_size();
  CHECK_GE(label_blob_num_, 1);
  CHECK_EQ(label_blob_num_, this->layer_param_.image_gt_data_param().field_h_size());
  CHECK_EQ(label_blob_num_, this->layer_param_.image_gt_data_param().field_w_size());
  for (int nn = 0; nn < label_blob_num_; nn++) {
    downsample_rates_.push_back(this->layer_param_.image_gt_data_param().downsample_rate(nn));
    field_ws_.push_back(this->layer_param_.image_gt_data_param().field_w(nn));
    field_hs_.push_back(this->layer_param_.image_gt_data_param().field_h(nn));
    int label_height = round(template_height/downsample_rates_[nn]); 
    int label_width = round(template_width/downsample_rates_[nn]);
    top[nn+1]->Reshape(batch_size, label_channel_, label_height, label_width);
    
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(batch_size, label_channel_, label_height, label_width);
      this->prefetch_[i].labels_.push_back(label_blob_pointer);
    }
    LOG(INFO) << "output label size "<<nn<<" : " << top[nn+1]->num() << ","
        << top[nn+1]->channels() << "," << top[nn+1]->height() << ","
        << top[nn+1]->width();
  }

  // setup for output gt boxes
  output_gt_boxes_ = this->layer_param_.image_gt_data_param().output_gt_boxes();
  if (output_gt_boxes_) {
    //dummy reshape
    top[label_blob_num_+1]->Reshape(1, 7, 1, 1);    
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(1, 7, 1, 1);
      this->prefetch_[i].labels_.push_back(label_blob_pointer);
    }
    LOG(INFO) << "output gt boxes size: " << top[label_blob_num_+1]->num() << ","
        << top[label_blob_num_+1]->channels() << "," << top[label_blob_num_+1]->height() 
        << "," << top[label_blob_num_+1]->width();
  }

  // data mean, only for mean values instead of mean file
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_values_) {
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
unsigned int ImageGtDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void ImageGtDataLayer<Dtype>::ShuffleList() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_list_.begin(), image_list_.end(), prefetch_rng);
}


template <typename Dtype>
void ImageGtDataLayer<Dtype>::BoundingboxAffine(vector<vector<float> >& bbs,
        float w_scale, float h_scale, float w_shift, float h_shift) {
  for (int ww = 0; ww < bbs.size(); ww++) {
    // rescale
    bbs[ww][ImageGtDataLayer<Dtype>::X1] = w_scale*bbs[ww][ImageGtDataLayer<Dtype>::X1];
    bbs[ww][ImageGtDataLayer<Dtype>::X2] = w_scale*bbs[ww][ImageGtDataLayer<Dtype>::X2];
    bbs[ww][ImageGtDataLayer<Dtype>::Y1] = h_scale*bbs[ww][ImageGtDataLayer<Dtype>::Y1];
    bbs[ww][ImageGtDataLayer<Dtype>::Y2] = h_scale*bbs[ww][ImageGtDataLayer<Dtype>::Y2];
    // shift
    bbs[ww][ImageGtDataLayer<Dtype>::X1] = bbs[ww][ImageGtDataLayer<Dtype>::X1]+w_shift;
    bbs[ww][ImageGtDataLayer<Dtype>::X2] = bbs[ww][ImageGtDataLayer<Dtype>::X2]+w_shift;
    bbs[ww][ImageGtDataLayer<Dtype>::Y1] = bbs[ww][ImageGtDataLayer<Dtype>::Y1]+h_shift;
    bbs[ww][ImageGtDataLayer<Dtype>::Y2] = bbs[ww][ImageGtDataLayer<Dtype>::Y2]+h_shift;
  }
}

// Thread fetching the data
template <typename Dtype>
void ImageGtDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  double label_time = 0;
  CPUTimer timer;
  Dtype* top_data = batch->data_.mutable_cpu_data();
  const int template_width = batch->data_.width();
  const int template_height = batch->data_.height();
  // zero out batch
  caffe_set(batch->data_.count(), Dtype(0), top_data);  

  const Dtype scale = this->layer_param_.image_gt_data_param().scale();
  const int batch_size = this->layer_param_.image_gt_data_param().batch_size();
  const bool mirror = this->transform_param_.mirror();
  const float fg_threshold = this->layer_param_.image_gt_data_param().fg_threshold();
  const float min_gt_width = this->layer_param_.image_gt_data_param().min_gt_width();
  const float min_gt_height = this->layer_param_.image_gt_data_param().min_gt_height();
  const bool do_multiple_scale = this->layer_param_.image_gt_data_param().do_multiple_scale()
                                 && this->layer_param_.image_gt_data_param().has_min_scale()
                                 && this->layer_param_.image_gt_data_param().has_max_scale();
  const bool whaspect = this->layer_param_.image_gt_data_param().has_min_whaspect()
                        && this->layer_param_.image_gt_data_param().has_max_whaspect();

  // label
  vector<Dtype*> top_labels; vector<int> label_counts;
  vector<int> label_spatial_dims;
  for (int nn = 0; nn < label_blob_num_; nn++) {
    top_labels.push_back(batch->labels_[nn]->mutable_cpu_data());
    label_counts.push_back(batch->labels_[nn]->count());
    label_spatial_dims.push_back(batch->labels_[nn]->width()*batch->labels_[nn]->height());
    // zero out batch
    caffe_set(label_counts[nn], Dtype(0), top_labels[nn]);
  }  
  
  // gt boxes
  vector<vector<Dtype> > gt_boxes;
  
  int item_id = 0; 
  int image_database_size = image_database_.size();
  CHECK_EQ(image_database_size,windows_.size());
  CHECK_EQ(image_database_size,roni_windows_.size());

  for (int dummy = 0; dummy < batch_size; ++dummy) {
      // sample a window
      timer.Start();
      CHECK_GT(image_database_size, list_id_);
      const unsigned int image_window_index = image_list_[list_id_];
      vector<vector<float> > windows = windows_[image_window_index];
      vector<vector<float> > roni_windows = roni_windows_[image_window_index];
      
      bool do_mirror = mirror && PrefetchRand() % 2;

      // load the image containing the window
      pair<std::string, vector<int> > image = image_database_[image_window_index];

      cv::Mat cv_img;
      if (this->cache_images_) {
        pair<std::string, Datum> image_cached = image_database_cache_[image_window_index];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      
      // horizontal flip at random
      int img_height = cv_img.rows, img_width = cv_img.cols;
      if (do_mirror) {
        cv::flip(cv_img, cv_img, 1);
        for (int ww = 0; ww < windows.size(); ++ww) {
          float tmp;
          windows[ww][ImageGtDataLayer<Dtype>::X1] = img_width-windows[ww][ImageGtDataLayer<Dtype>::X1];
          windows[ww][ImageGtDataLayer<Dtype>::X2] = img_width-windows[ww][ImageGtDataLayer<Dtype>::X2];
          tmp = windows[ww][ImageGtDataLayer<Dtype>::X1];
          windows[ww][ImageGtDataLayer<Dtype>::X1] = windows[ww][ImageGtDataLayer<Dtype>::X2];
          windows[ww][ImageGtDataLayer<Dtype>::X2] = tmp;
        }
        
        for (int ww = 0; ww < roni_windows.size(); ++ww) {
          float tmp;
          roni_windows[ww][ImageGtDataLayer<Dtype>::X1] = img_width-roni_windows[ww][ImageGtDataLayer<Dtype>::X1];
          roni_windows[ww][ImageGtDataLayer<Dtype>::X2] = img_width-roni_windows[ww][ImageGtDataLayer<Dtype>::X2];
          tmp = roni_windows[ww][ImageGtDataLayer<Dtype>::X1];
          roni_windows[ww][ImageGtDataLayer<Dtype>::X1] = roni_windows[ww][ImageGtDataLayer<Dtype>::X2];
          roni_windows[ww][ImageGtDataLayer<Dtype>::X2] = tmp;
        }
      }
      // resize image if needed
      if (this->layer_param_.image_gt_data_param().has_resize_width()
          && this->layer_param_.image_gt_data_param().has_resize_height()) {
        int resize_width = this->layer_param_.image_gt_data_param().resize_width();
        int resize_height = this->layer_param_.image_gt_data_param().resize_height();
        if (resize_width != img_width || resize_height != img_height) {
          float width_factor = float(resize_width)/img_width;
          float height_factor = float(resize_height)/img_height;
          cv::Size cv_resize_size;
          cv_resize_size.width = resize_width; cv_resize_size.height = resize_height;
          cv::resize(cv_img, cv_img, cv_resize_size, 0, 0, cv::INTER_LINEAR);
          // resize bounding boxes
          BoundingboxAffine(windows,width_factor,height_factor,0,0);
          BoundingboxAffine(roni_windows,width_factor,height_factor,0,0);
        }
      }

      img_height = cv_img.rows, img_width = cv_img.cols;
      CHECK_LE(template_width, img_width); CHECK_LE(template_height, img_height);
      int src_offset_x=0, src_offset_y=0, dst_offset_x=0, dst_offset_y=0;
      int copy_width = template_width, copy_height = template_height;
      float width_rescale_factor  = 1, height_rescale_factor = 1;
      int sel_id=-1; float sel_center_x, sel_center_y;
      // decide crop center
      if (windows.size() != 0) {
        sel_id = PrefetchRand() % windows.size();
        sel_center_x = (windows[sel_id][ImageGtDataLayer<Dtype>::X1]+windows[sel_id][ImageGtDataLayer<Dtype>::X2])/2.0;
        sel_center_y = (windows[sel_id][ImageGtDataLayer<Dtype>::Y1]+windows[sel_id][ImageGtDataLayer<Dtype>::Y2])/2.0;
      } else {
        sel_center_x = PrefetchRand() % (img_width-template_width+1) + template_width/2.0;
        sel_center_y = PrefetchRand() % (img_height-template_height+1) + template_height/2.0;
      }

      // multiple scaling
      if (do_multiple_scale && windows.size()!=0 && PrefetchRand()%2) {
        float octave = 2; 
        int min_scale = this->layer_param_.image_gt_data_param().min_scale();
        int max_scale = this->layer_param_.image_gt_data_param().max_scale();
        float scale_order = log2(float(max_scale)/float(min_scale));
        int num_scale = round(octave*scale_order+1); 
        vector<float> resize_scales(num_scale);
        for (int ss = 0; ss < resize_scales.size(); ss++) resize_scales[ss]=ss/octave;  
        float X1 = windows[sel_id][ImageGtDataLayer<Dtype>::X1];
        float X2 = windows[sel_id][ImageGtDataLayer<Dtype>::X2];
        float Y1 = windows[sel_id][ImageGtDataLayer<Dtype>::Y1];
        float Y2 = windows[sel_id][ImageGtDataLayer<Dtype>::Y2];
        float bb_width = X2-X1, bb_height = Y2-Y1;
        float max_iou = 0; float match_scale = 0;
        for (int ss = 0; ss < resize_scales.size(); ss++) {
          float bb_area = bb_width*bb_height;
          float field_area = min_scale*min_scale*pow(2,resize_scales[ss])*pow(2,resize_scales[ss]); 
          float tmp_iou = std::min(bb_area,field_area) / std::max(bb_area,field_area);
          if (tmp_iou > max_iou) {
            max_iou = tmp_iou; match_scale = resize_scales[ss];
          }
        }
        for (int ss = 0; ss < resize_scales.size(); ss++) resize_scales[ss] -= match_scale;
        //randomly select a scale to reisize
        int random_idx = PrefetchRand() % resize_scales.size();
        float random_scale = resize_scales[random_idx];
        float rescale_factor = pow(2,random_scale);
        width_rescale_factor = rescale_factor; height_rescale_factor = rescale_factor;
        bool do_whaspect = whaspect && PrefetchRand() % 2;
        // aspect ratio scaling
        if (do_whaspect) {
          float min_whaspect = this->layer_param_.image_gt_data_param().min_whaspect();
          float max_whaspect = this->layer_param_.image_gt_data_param().max_whaspect();
          float whaspect_interval = 0.05f;
          int intervals = round((max_whaspect-min_whaspect) / whaspect_interval);
          float random_aspect = (PrefetchRand() % intervals) * whaspect_interval + min_whaspect;
          float aspect_multiplier = random_aspect / (bb_width/bb_height);
          bool resize_width_flag = PrefetchRand() % 2;
          if (resize_width_flag) {
            float target_width = bb_width*width_rescale_factor*aspect_multiplier;
            if (target_width >= min_scale*0.8 && target_width <= max_scale*1.2) {
              width_rescale_factor *= aspect_multiplier;
            }
          } else {
            float target_height = bb_height*height_rescale_factor/aspect_multiplier;
            if (target_height >= min_scale*0.8 && target_height <= max_scale*1.2) {
              height_rescale_factor /= aspect_multiplier;
            }
          }
        }
        float final_aspect = bb_width*width_rescale_factor/(bb_height*height_rescale_factor);
        DLOG(INFO)<<"bb_width: "<<bb_width<<", bb_height: "<<bb_height<<", match_scale: "<<match_scale
                 <<", random_idx: "<<random_idx<<", random_scale: "<<random_scale<<", rescale_factor: "<<rescale_factor
                 <<", width_scale: "<<width_rescale_factor<<", height_scale: "
                 <<height_rescale_factor<<", final_aspect: "<<final_aspect;
      }

      int rescale_height = round(img_height*height_rescale_factor);
      int rescale_width = round(img_width*width_rescale_factor);
      if (width_rescale_factor != 1 || height_rescale_factor != 1) {
        // if upsampling is too large, crop the image first, then upsample
        if (width_rescale_factor>1.5 || height_rescale_factor>1.5) {
          int crop_w = round(1.2*img_width/width_rescale_factor);
          int crop_h = round(1.2*img_height/height_rescale_factor);
          crop_w = std::min(crop_w,img_width);
          crop_h = std::min(crop_h,img_height);
          int crop_x1 = round(sel_center_x-crop_w*0.5); 
          int crop_y1 = round(sel_center_y-crop_h*0.5); 
          crop_x1 = std::max(crop_x1,0); crop_y1 = std::max(crop_y1,0);
          int diff_x = std::max(crop_x1+crop_w-img_width,0); crop_x1 -= diff_x;
          int diff_y = std::max(crop_y1+crop_h-img_height,0); crop_y1 -= diff_y;
          CHECK_GE(crop_x1,0); CHECK_GE(crop_y1,0);
          // crop image
          cv::Rect roi(crop_x1, crop_y1, crop_w, crop_h);
          cv_img = cv_img(roi);
          //shift center coordinates
          sel_center_x -= crop_x1; sel_center_y -= crop_y1;
          //shift bounding boxes
          BoundingboxAffine(windows,1,1,-crop_x1,-crop_y1);
          BoundingboxAffine(roni_windows,1,1,-crop_x1,-crop_y1);
          rescale_width = round(cv_img.cols*width_rescale_factor); 
          rescale_height = round(cv_img.rows*height_rescale_factor);
        }
        cv::Size cv_rescale_size;
        cv_rescale_size.width = rescale_width; cv_rescale_size.height = rescale_height;
        cv::resize(cv_img, cv_img, cv_rescale_size, 0, 0, cv::INTER_LINEAR);
        img_height = cv_img.rows, img_width = cv_img.cols;
      }
        
      // resize bounding boxes
      BoundingboxAffine(windows,width_rescale_factor,height_rescale_factor,0,0);
      BoundingboxAffine(roni_windows,width_rescale_factor,height_rescale_factor,0,0);

      int noise_x = PrefetchRand() % 20 - 10, noise_y = PrefetchRand() % 20 - 10;   
      if (rescale_width < template_width) {
        dst_offset_x = 0; copy_width = rescale_width; 
        src_offset_x = round((template_width-rescale_width)/2.0) + noise_x;
        src_offset_x = std::max(0,src_offset_x);
        src_offset_x = std::min(template_width-rescale_width,src_offset_x);
      } else if (rescale_width > template_width) {
        src_offset_x = 0; copy_width = template_width;
        int center_x = round(sel_center_x*width_rescale_factor)+noise_x;
        dst_offset_x = center_x-round(template_width/2.0);
        dst_offset_x = std::max(0,dst_offset_x);
        dst_offset_x = std::min(rescale_width-template_width,dst_offset_x);
      } else {
        src_offset_x = 0; dst_offset_x = 0; copy_width = template_width; 
      }
     
      if (rescale_height < template_height) {
        dst_offset_y = 0; copy_height = rescale_height; 
        src_offset_y = round((template_height-rescale_height)/2.0) + noise_y;
        src_offset_y = std::max(0,src_offset_y);
        src_offset_y = std::min(template_height-rescale_height,src_offset_y);
      } else if (rescale_height > template_height) {
        src_offset_y = 0; copy_height = template_height;
        int center_y = round(sel_center_y*height_rescale_factor)+noise_y;
        dst_offset_y = center_y-round(template_height/2.0);
        dst_offset_y = std::max(0,dst_offset_y);
        dst_offset_y = std::min(rescale_height-template_height,dst_offset_y);
      } else {
        src_offset_y = 0; dst_offset_y = 0; copy_height = template_height;
      }
        
      // shift bounding boxes
      BoundingboxAffine(windows,1,1,src_offset_x-dst_offset_x,src_offset_y-dst_offset_y);
      BoundingboxAffine(roni_windows,1,1,src_offset_x-dst_offset_x,src_offset_y-dst_offset_y);

      // copy the original image into top_data
      const int channels = cv_img.channels();     
      CHECK_LE(copy_width,template_width); CHECK_LE(copy_height,template_height);
      CHECK_LE(copy_width,img_width); CHECK_LE(copy_height,img_height);
      for (int h = src_offset_y; h < src_offset_y+copy_height; ++h) {
        const uchar* ptr = cv_img.ptr<uchar>(h-src_offset_y+dst_offset_y);
        for (int w = src_offset_x; w < src_offset_x+copy_width; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id * channels + c) * template_height + h) * template_width + w;
            int img_index = (w-src_offset_x+dst_offset_x) * channels + c;
            Dtype pixel = static_cast<Dtype>(ptr[img_index]);
            if (this->has_mean_values_) {
              top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
            } else {
              top_data[top_index] = pixel * scale;
            }
          }
        }
      }
      trans_time += timer.MicroSeconds();
      timer.Start();
      
      // get window label
      CHECK_EQ(label_channel_,6);
      vector<vector<Dtype> > gts;
      vector<int> match_times(windows.size());
      vector<int> max_bb_nn(windows.size());
      vector<int> max_bb_index(windows.size());
      vector<Dtype> max_bb_iou(windows.size());
      for (int ww = 0; ww < windows.size(); ++ww) {
        Dtype x1 = windows[ww][ImageGtDataLayer<Dtype>::X1];
        Dtype y1 = windows[ww][ImageGtDataLayer<Dtype>::Y1];
        Dtype x2 = windows[ww][ImageGtDataLayer<Dtype>::X2];
        Dtype y2 = windows[ww][ImageGtDataLayer<Dtype>::Y2];  
        //filter those gt bounding boxes whose centers are outside of the image
        Dtype xc = (x1+x2)/2.0, yc = (y1+y2)/2.0;
        if (xc < 0 || xc >= template_width || yc < 0 || yc >= template_height) {
          windows[ww][ImageGtDataLayer<Dtype>::IGNORE] = 1;
        }
        //filter gt bbs smaller than minimum size
        if ((x2-x1+1) < min_gt_width) {
          windows[ww][ImageGtDataLayer<Dtype>::IGNORE] = 1;
        }
        if ((y2-y1+1) < min_gt_height) {
          windows[ww][ImageGtDataLayer<Dtype>::IGNORE] = 1;
        }
        
        vector<Dtype> gt(5);
        gt[0] = x1; gt[1] = y1; gt[2] = x2; gt[3] = y2; 
        gt[4] = windows[ww][ImageGtDataLayer<Dtype>::LABEL];
        gts.push_back(gt);
        
        // for gt boxes
        vector<Dtype> gt_box(7);
        gt_box[0] = dummy; gt_box[1] = x1; gt_box[2] = y1; gt_box[3] = x2; gt_box[4] = y2;
        gt_box[5] = windows[ww][ImageGtDataLayer<Dtype>::LABEL]; 
        gt_box[6] = windows[ww][ImageGtDataLayer<Dtype>::IGNORE];
        gt_boxes.push_back(gt_box);
      }

      vector<vector<Dtype> > ronis;
      for (int ww = 0; ww < roni_windows.size(); ++ww) {
        vector<Dtype> roni(4);
        roni[0] = roni_windows[ww][ImageGtDataLayer<Dtype>::X1];
        roni[1] = roni_windows[ww][ImageGtDataLayer<Dtype>::Y1];
        roni[2] = roni_windows[ww][ImageGtDataLayer<Dtype>::X2];
        roni[3] = roni_windows[ww][ImageGtDataLayer<Dtype>::Y2];            
        ronis.push_back(roni);
      }

      // label data transfer
      for (int nn = 0; nn < label_blob_num_; nn++) {
        const int label_height = round(template_height/float(downsample_rates_[nn]));
        const int label_width = round(template_width/float(downsample_rates_[nn]));
        const int label_offset_x = round(src_offset_x/float(downsample_rates_[nn]));
        const int label_offset_y = round(src_offset_y/float(downsample_rates_[nn]));
        const int label_copy_width = round(copy_width/float(downsample_rates_[nn]));
        const int label_copy_height = round(copy_height/float(downsample_rates_[nn]));
        const int spatial_dim = label_height*label_width;
        CHECK_EQ(spatial_dim,label_spatial_dims[nn]);
        const Dtype radius_w = field_ws_[nn] / Dtype(2);
        const Dtype radius_h = field_hs_[nn] / Dtype(2);

        for (int h = 0; h < label_height; ++h)
          for (int w = 0; w < label_width; ++w) {
            int top_index = (item_id * label_channel_ * label_height + h) * label_width + w;
            if (w < label_offset_x || w >= label_offset_x+label_copy_width
                || h < label_offset_y || h >= label_offset_y+label_copy_height) {
              top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
              continue;
            }
            Dtype xx1, yy1, xx2, yy2;
            xx1 = (w+0.5)*downsample_rates_[nn]-radius_w; 
            xx2 = (w+0.5)*downsample_rates_[nn]+radius_w;
            yy1 = (h+0.5)*downsample_rates_[nn]-radius_h; 
            yy2 = (h+0.5)*downsample_rates_[nn]+radius_h;
            
            // decide if the candidate bb is in the RONI
            float sum_iou = 0;
            for (int ww = 0; ww < ronis.size(); ++ww) {
              Dtype iou = BoxIOU(xx1, yy1, xx2-xx1, yy2-yy1, ronis[ww][0], ronis[ww][1], 
                                 ronis[ww][2]-ronis[ww][0], ronis[ww][3]-ronis[ww][1], "IOFU");
              sum_iou += iou;
            }
            if (sum_iou >= 0.4) {
              top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
              continue;
            }

            bool flag = false; int match_idx;
            float max_iou = 0;
            for (int ww = 0; ww < gts.size(); ++ww) {
              Dtype iou = BoxIOU(gts[ww][0], gts[ww][1] ,gts[ww][2]-gts[ww][0], 
                                 gts[ww][3]-gts[ww][1], xx1, yy1, xx2-xx1, yy2-yy1, "IOU");
              if (iou > max_iou) {
                flag = true; match_idx = ww; max_iou = iou;
              }
              if (iou > max_bb_iou[ww]) {
                max_bb_iou[ww] = iou; max_bb_nn[ww] = nn; max_bb_index[ww] = top_index;
              }
            }  
            if (flag && max_iou > fg_threshold) {
              float x1 = windows[match_idx][ImageGtDataLayer<Dtype>::X1];
              float y1 = windows[match_idx][ImageGtDataLayer<Dtype>::Y1];
              float x2 = windows[match_idx][ImageGtDataLayer<Dtype>::X2];
              float y2 = windows[match_idx][ImageGtDataLayer<Dtype>::Y2];
              int ignore = windows[match_idx][ImageGtDataLayer<Dtype>::IGNORE];
              if (ignore == 0) {
                top_labels[nn][top_index] = windows[match_idx][ImageGtDataLayer<Dtype>::LABEL];
              } else {
                top_labels[nn][top_index] = 0;
              }
              top_labels[nn][top_index+spatial_dim] = (x1+x2)/Dtype(2.0);
              top_labels[nn][top_index+2*spatial_dim] = (y1+y2)/Dtype(2.0);
              top_labels[nn][top_index+3*spatial_dim] = x2-x1;
              top_labels[nn][top_index+4*spatial_dim] = y2-y1;
              match_times[match_idx]++;
            } 
            // iou (if label==0&&iou>fg_threshold, it will be ignored)
            top_labels[nn][top_index+5*spatial_dim] = max_iou;
          }
      }

      //pick up those gt bounding boxes that are not matched yet
      for (int ww = 0; ww < gts.size(); ++ww) {
        if (windows[ww][ImageGtDataLayer<Dtype>::IGNORE] == 0 && match_times[ww] <= 0){ 
          // if iou is too small, still ignored
          if (max_bb_iou[ww] < 0.2) {
            continue;
          }
          int miss_nn = max_bb_nn[ww];
          const int label_height = round(template_height/float(downsample_rates_[miss_nn]));
          const int label_width = round(template_width/float(downsample_rates_[miss_nn]));
          const int spatial_dim = label_height*label_width;

          const float x1 = windows[ww][ImageGtDataLayer<Dtype>::X1];
          const float y1 = windows[ww][ImageGtDataLayer<Dtype>::Y1];
          const float x2 = windows[ww][ImageGtDataLayer<Dtype>::X2];
          const float y2 = windows[ww][ImageGtDataLayer<Dtype>::Y2];
          Dtype xc = (x1+x2)/Dtype(2.0), yc = (y1+y2)/Dtype(2.0);
          int hc = floor(yc/downsample_rates_[miss_nn]);
          hc = std::max(0,hc); hc = std::min(label_height-1,hc);
          int wc = floor(xc/downsample_rates_[miss_nn]);
          wc = std::max(0,wc); wc = std::min(label_width-1,wc);
          const int miss_index = (item_id * label_channel_ * label_height + hc) * label_width + wc;
          if (top_labels[miss_nn][miss_index] > 0) {
            continue; 
          }
          top_labels[miss_nn][miss_index] = windows[ww][ImageGtDataLayer<Dtype>::LABEL];
          top_labels[miss_nn][miss_index+spatial_dim] = xc; 
          top_labels[miss_nn][miss_index+2*spatial_dim] = yc;
          top_labels[miss_nn][miss_index+3*spatial_dim] = x2-x1;
          top_labels[miss_nn][miss_index+4*spatial_dim] = y2-y1;
        }
      }
    
      label_time += timer.MicroSeconds();
      #if 0
      // useful debugging code for dumping data to disk
      string file_id;
      std::stringstream ss;
      ss << list_id_;
      ss >> file_id;
      string root_dir = string("examples/dump/");
      string outputstr = root_dir + file_id + string("_info.txt");
      std::ofstream inf(outputstr.c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << do_mirror << std::endl;
      for (int ww = 0; ww < windows.size(); ++ww) {
        inf << windows[ww][ImageGtDataLayer<Dtype>::LABEL] <<", "
            << windows[ww][ImageGtDataLayer<Dtype>::IGNORE] <<", "
            << windows[ww][ImageGtDataLayer<Dtype>::X1] <<", "
            << windows[ww][ImageGtDataLayer<Dtype>::Y1] <<", "
            << windows[ww][ImageGtDataLayer<Dtype>::X2] <<", "
            << windows[ww][ImageGtDataLayer<Dtype>::Y2] << std::endl;
      }
      for (int ww = 0; ww < roni_windows.size(); ++ww) {
        inf << 100 << ", "<< 100 << ", "  
            << roni_windows[ww][ImageGtDataLayer<Dtype>::X1] <<", "
            << roni_windows[ww][ImageGtDataLayer<Dtype>::Y1] <<", "
            << roni_windows[ww][ImageGtDataLayer<Dtype>::X2] <<", "
            << roni_windows[ww][ImageGtDataLayer<Dtype>::Y2] << std::endl;
      }
      inf.close();
      
      std::ofstream top_data_file((root_dir + file_id + string("_data.txt")).c_str(),
          std::ofstream::out);
      for (int c = 0; c < channels; ++c) {
        for (int w = 0; w < template_width; ++w) {
          for (int h = 0; h < template_height; ++h) {
            top_data_file << top_data[((item_id * channels + c) * template_height + h)
                          * template_width + w]<<std::endl;
          }
        }
      }
      top_data_file.close();
      
      for (int nn = 0; nn < label_blob_num_; nn++) {
        const int label_height = round(template_height/float(downsample_rates_[nn]));
        const int label_width = round(template_width/float(downsample_rates_[nn]));
        string label_id; std::stringstream sss;
        sss << nn; sss >> label_id;
        std::ofstream top_label_file((root_dir + file_id + string("_")+label_id+string("_label.txt")).c_str(),
            std::ofstream::out);
        for (int k = 0; k < label_channel_; k++) {
          for (int w = 0; w < label_width; ++w) {
            for (int h = 0; h < label_height; ++h) {
              top_label_file << top_labels[nn][((item_id * label_channel_ + k) * label_height + h) 
                             * label_width + w]<<std::endl;
            }
          }
        }
        top_label_file.close();
      }

      #endif

      item_id++;
      list_id_++;
      if (list_id_ >= image_database_size) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        list_id_ = 0;
        if (this->layer_param_.image_gt_data_param().shuffle()) {
          LOG(INFO) << "Restarting shuffling data.";
          ShuffleList();
       }
      }
  }
  
  // output gt boxes [img_id, x1, y1, x2, y2, label, ignored] for detection subnet
  if (output_gt_boxes_) {
    int num_gt_boxes = gt_boxes.size();
    const int gt_dim = 7;
    // for special case when there is no gt
    if (num_gt_boxes <= 0) {
      batch->labels_[label_blob_num_]->Reshape(1, gt_dim, 1, 1);
      Dtype* gt_boxes_data = batch->labels_[label_blob_num_]->mutable_cpu_data();
      gt_boxes_data[0]=0; gt_boxes_data[1]=1; gt_boxes_data[2]=1; gt_boxes_data[3]=2;
      gt_boxes_data[4]=2; gt_boxes_data[5]=1; gt_boxes_data[6]=1;
    } else {
      batch->labels_[label_blob_num_]->Reshape(num_gt_boxes, gt_dim, 1, 1);
      Dtype* gt_boxes_data = batch->labels_[label_blob_num_]->mutable_cpu_data();
      for (int i = 0; i < num_gt_boxes; i++) {
        for (int j = 0; j < gt_dim; j++) {
          gt_boxes_data[i*gt_dim+j] = gt_boxes[i][j];
        }
      }
    }
  }
  
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  DLOG(INFO) << "Label time: " << label_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageGtDataLayer);
REGISTER_LAYER_CLASS(ImageGtData);

}  // namespace caffe
#endif  // USE_OPENCV
