## A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection

by Zhaowei Cai, Quanfu Fan, Rogerio Feris and Nuno Vasconcelos

This implementation is written by Zhaowei Cai at UC San Diego.

### Introduction

MS-CNN is a unified multi-scale object detection framework based on deep convolutional networks, which includes an object proposal sub-network and an object detection sub-network. The unified network can be trained altogether end-to-end. 

### Citations

If you use our code/model/data, please cite our paper:

    @inproceedings{cai16mscnn,
      author = {Zhaowei Cai and Quanfu Fan and Rogerio Feris and Nuno Vasconcelos},
      Title = {A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection},
      booktitle = {ECCV},
      Year  = {2016}
    }

### Updates

This repository is merged to the latest Caffe. There is very minor numerical difference from the old version. By using the latest vresions of Caffe, CUDA and cuDNN, the speeds could be doubled. If you want to use the old version of code, you can download it from [MSCNN-V1.0](http://www.svcl.ucsd.edu/projects/mscnn/mscnn_v1.0.zip). 

### Requirements

1. cuDNN is required to avoid the issue of out-of-memory and have the same running speed described in our paper. For now, CUDA 8.0 with cuDNN v5 is tested. The other versions should be working.

2. If you want to use our MATLAB scripts to run the detection demo, caffe MATLAB wrapper is required. Please build matcaffe before running the detection demo. 

3. This code has been tested on Ubuntu 14.04 with an NVIDIA Titan GPU.

### Installation

1. Clone the MS-CNN repository, and we'll call the directory that you cloned MS-CNN into `MSCNN_ROOT`
    ```Shell
    git clone https://github.com/zhaoweicai/mscnn.git
    ```
  
2. Build MS-CNN
    ```Shell
    cd $MSCNN_ROOT/
    # Follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j 16

    # If you want to use MSCNN detection demo, build MATLAB wrapper as well
    make matcaffe
    ```

### Training MS-CNN (KITTI car)

1. Set up KITTI dataset by yourself.

2. Get the training data for KITTI
    ```Shell
    cd $MSCNN_ROOT/data/
    sh get_kitti_data.sh
    ```
    
    This will download train/val split image lists for the experiments, and window files for training/finetuning MS-CNN models. You can also use the provided MATLAB scripts `mscnn_kitti_car_window_file.m` under `$MSCNN_ROOT/data/kitti/` to generate your own window files. If you use the provided window files, replace `/your/KITTI/path/` in the files to your KITTI path.

3. Download VGG16 from [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), and put it into `$MSCNN_ROOT/models/VGG/`.

4. Now you can start to train MS-CNN models. Multiple shell scripts are provided to train different models described in our paper. We take `mscnn-7s-576-2x` for example. 
    ```Shell
    cd $MSCNN_ROOT/examples/kitti_car/mscnn-7s-576-2x/
    sh train_mscnn.sh
    ```
   As described in the paper, the training process is split into two steps. Usually the first step can be shared by different models if you only have modifications on detection sub-network. For example, the first training step can be shared by `mscnn-7s-576-2x` and `mscnn-7s-576`. Meanwhile, log files will be generated along the training procedures. 
 

### Pretrained model (KITTI car)

Download pre-trained MS-CNN models
```Shell
cd $MSCNN_ROOT/examples/kitti_car/
sh fetch_mscnn_car_model.sh
``` 
This will download the pretrained model for KITTI car into `$MSCNN_ROOT/examples/kitti_car/mscnn-8s-768-trainval-pretrained/`. You can produce exactly the same results as described in our paper with these pretrained models.

### Testing Demo (KITTI car)

Once the pretrained models or models trained by yourself are available, you can use the MATLAB script `run_mscnn_detection.m` under `$MSCNN_ROOT/examples/kitti_car/` to obtain the detection and proposal results. Set the right dataset path and choose the model that you want to test in the demo script. The default setting is to test the pretrained model. The final results will be saved as .txt files.

### KITTI Evaluation

Compile `evaluate_object.cpp` under `$MSCNN_ROOT/examples/kitti_result/eval/` by yourself. Use `writeDetForEval.m` under `$MSCNN_ROOT/examples/kitti_result/` to transform the detection results into KITTI data format and evaluate the detection performance. Remember to change the corresponding directories in the evaluation script. 

### Disclaimer

1. The CPU version is not fully tested. The GPU version is strongly recommended.
 
2. Since some changes have been made after ECCV submission, you may not have exactly the same results in the paper by training your own models. But you should have equivelant performance. 

3. Since the numbers of training samples vary vastly for different classes, the model robustness varies too (car>ped>cyc).

4. Although the final results we submitted were from model `mscnn-8s-768-trainval`, our later experiments have shown that `mscnn-7s-576-2x-trainval` can achieve even better performance for car, and 2x faster speed. For ped/cyc however, the performance decreases due to the much less training instances.  

5. If the training does not converge or the performance is very bad, try some other random seeds. You should obtain fair performance after a few tries. Due to the randomness, you cann't fully reproduce the same models, but the performance should be close.

If you encounter any issue when using our code or model, please let me know.
