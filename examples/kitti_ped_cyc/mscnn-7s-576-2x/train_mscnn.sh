
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1st.prototxt \
  --weights=../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel \
  --gpu=0  2>&1 | tee log_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_kitti_train_1st_iter_10000.caffemodel \
  --gpu=0  2>&1 | tee log_2nd.txt
