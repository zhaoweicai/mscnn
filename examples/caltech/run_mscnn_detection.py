# Modified from "https://raw.githubusercontent.com/GBJim/mscnn/master/examples/caltech/run_mscnn_detection.py"
# Modified by shls

from __future__ import division
import os
import math
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from nms.gpu_nms import gpu_nms
import sys
import glob
import cv2
import argparse
import re
import time
from scipy.misc import imread

# set caffe root and lib   
caffe_root = '~/mscaffe/'
sys.path.insert(0, caffe_root + "install/python")
sys.path.insert(0, caffe_root + "lib")
import caffe

CALTECH_DATA_PATH = "/root/caltech/data/"
IMG_PATH = os.path.join(CALTECH_DATA_PATH + "images")


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a MSCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='prototxt',
                        help='prototxt file defining the network',
                        default='examples/caltech/mscnn-7s-720-pretrained/mscnn_deploy.prototxt', type=str)
    parser.add_argument('--weights', dest='caffemodel',
                        help='model to test',
                        default='examples/caltech/mscnn-7s-720-pretrained/mscnn_caltech_train_2nd_iter_20000.caffemodel'\
                        , type=str)

    parser.add_argument('--do_bb_norm', dest='do_bb_norm',help="Whether to denormalize the box with std or means.\
    Author's pretrained model does not need this. ",
                default=True , type=bool)
    parser.add_argument('--height', dest='height',help="Decide the resizing height of input model and images",
                default=720 , type=int)
    parser.add_argument('--detection', dest='dt_name',  help='model to test', default='detection_1', type=str)
    parser.add_argument('--video_file', dest='video_name',  help='video to test', default='', type=str)

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
   

def filter_proposals(proposals, threshold=-10):
    #Bug 1 Fixed
    keeps = (proposals[:, -1] >= threshold) & (proposals[:, 2] != 0) & (proposals[:, 3] != 0)
    return keeps


def im_normalize(im, target_size, mu=[104, 117, 123] ):
    n_im = cv2.resize(im, target_size).astype(np.float32)
    
    #Substracts mu from testing-BGR image
    n_im -= mu
    #print(im.shape)
    n_im = np.swapaxes(n_im, 1,2)
    n_im = np.swapaxes(n_im, 0,1)
    n_im = np.array([n_im])
    #print(n_im.shape)
    #print(n_im.shape)
    return n_im


def bbox_denormalize(bbox_pred, proposals, ratios, orgW, orgH):
    
    bbox_means = [0, 0, 0, 0]
    bbox_stds = [0.1, 0.1, 0.2, 0.2]

    if args.do_bb_norm:
        bbox_pred *= bbox_stds 
        bbox_pred += bbox_means

    ctr_x = proposals[:,0]+0.5*proposals[:,2]
    ctr_y = proposals[:,1]+0.5*proposals[:,3]

    tx = bbox_pred[:,0] *proposals[:,2] + ctr_x
    ty = bbox_pred[:,1] *proposals[:,3] + ctr_y

    tw = proposals[:,2] * np.exp(bbox_pred[:,2])
    th = proposals[:,3] * np.exp(bbox_pred[:,3])

    #Fix Bug 2
    tx -= tw/2 
    ty -= th/2
    tx /= ratios[0] 
    tw /= ratios[0]
    ty /= ratios[1] 
    th /= ratios[1]

    tx[tx < 0] = 0
    ty[ty < 0] = 0
    #Fix Bug 3
    tw[tw > (orgW - tx)] = (orgW - tx[tw > (orgW - tx)])
    th[th > (orgH - ty)] = (orgH - ty[th > (orgH - ty)])
    new_boxes = np.hstack((tx[:, None], ty[:, None], tw[:, None], th[:, None])).astype(np.float32).reshape((-1, 4)) #suspecious
    return new_boxes


def get_confidence(cls_pred):
    exp_score = np.exp(cls_pred)
    sum_exp_score = np.sum(exp_score, 1)
    confidence = exp_score[:, 1] / sum_exp_score
    
    return confidence

#mu is the mean of BGR 
# im_dect use file path
# def im_detect(net, file_path, target_size= (960, 720)):

#     im = cv2.imread(file_path)

# im_detect use im
def im_detect(net, im, target_size= (960, 720)):
    orgH, orgW, _ = im.shape
    ratios = (target_size[0]/orgW, (target_size[1]/orgH ))
    im = im_normalize(im, target_size)
    
    #Feedforward
    net.blobs['data'].data[...] = im 
    output = net.forward()
    
    bbox_pred = output['bbox_pred']
    proposals = output['proposals_score'].reshape((-1,6))[:,1:]  #suspecious
    
    proposals[:,2] -=   proposals[:,0]
    proposals[:,3] -=   proposals[:,1]
    cls_pred = output['cls_pred']
    
    
    keeps = filter_proposals(proposals)
    bbox_pred =  bbox_pred[keeps]
    cls_pred = cls_pred[keeps]
    proposals = proposals[keeps]
    
    pedestrian_boxes = bbox_pred[:,4:8]
    boxes = bbox_denormalize(pedestrian_boxes, proposals, ratios, orgW, orgH)

    #Denormalize the confidence 
    
    confidence = get_confidence(cls_pred)
    return confidence, boxes

def nms(dets, thresh):
    
    if dets.shape[0] == 0:
        return []
    new_dets = np.copy(dets)
    new_dets[:,2] += new_dets[:,0]
    new_dets[:,3] += new_dets[:,1]
   
    return gpu_nms(new_dets, thresh, device_id=GPU_ID)
  
def write_caltech_results_file(net):
    # The follwing nested fucntions are for smart sorting
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    def insert_frame(target_frames, file_path,start_frame=29, frame_rate=30):
        file_name = file_path.split("/")[-1]
        set_num, v_num, frame_num = file_name[:-4].split("_")
        if int(frame_num) >= start_frame and int(frame_num) % frame_rate == 29:
            target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
            return 1
        else:
            return 0

    def detect(file_path,  NMS_THRESH = 0.3):
        if args.height == 720:
            target_size = (960, 720)
        elif args.height == 480:
            target_size = (640, 480)

        confidence, bboxes = im_detect(net, file_path, target_size)
    
        dets = np.hstack((bboxes,confidence[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        print("{} Bboxes".format(len(keep)))
        return dets[keep, :]


    def get_target_frames(image_set_list,  image_path):
        target_frames = {}
        total_frames = 0 
        for set_num in image_set_list:
            file_pattern = "{}/set{}_V*".format(image_path,set_num)
            print(file_pattern)
            file_list = sorted(glob.glob(file_pattern), key=natural_keys)
            for file_path in file_list:
                total_frames += insert_frame(target_frames, file_path)

        return target_frames, total_frames 
    
    

    def detection_to_file(target_path, v_num, file_list, detect,total_frames, current_frames, max_proposal=100, thresh=0):
        timer = Timer()
        w = open("{}/{}.txt".format(target_path, v_num), "w")
        for file_index, file_path in enumerate(file_list):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            frame_num = str(int(frame_num) +1)

            timer.tic()
            dets = detect(file_path)

            timer.toc()

            print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                   file_name ,current_frames+file_index+1 , total_frames))


            inds = np.where(dets[:, -1] >= thresh)[0]     
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                
                
                #Fix bug 6
                x = bbox[0]
                y = bbox[1] 
                width = bbox[2] 
                length =  bbox[3]
                if score*100 > 70:
                    print("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))
                    
                w.write("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))


        w.close()
        print("Evalutaion file {} has been writen".format(w.name))   
        return file_index + 1




    OUTPUT_PATH = os.path.join("./output",  DETECTION_NAME)
    if not os.path.exists(OUTPUT_PATH ):
        os.makedirs(OUTPUT_PATH )       

    image_set_list = ["06", "07" , "08", "09", "10"]
    target_frames, total_frames = get_target_frames(image_set_list,  IMG_PATH)

    current_frames = 0
    for set_num in target_frames:
        target_path = os.path.join(OUTPUT_PATH , set_num)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for v_num, file_list in target_frames[set_num].items():
            current_frames += detection_to_file(target_path, v_num, file_list, detect, total_frames, current_frames)

def video_prediction(net, video_name):
    vidcap = cv2.VideoCapture(video_name)
    success,im = vidcap.read()
    while success:
        confidence, boxes = im_dect(net, im)
        print confidence
        print boxes
        success,im = vidcap.read()
                
if __name__ == "__main__":
    args = parse_args()
    global GPU_ID
    global DETECTION_NAME
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    GPU_ID = args.gpu_id
    
    DETECTION_NAME = args.dt_name
    
    print("Loading Network")
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    print("MC-CNN model loaded")
    # Detect the video
    video_prediction(net,video_name)


    # Detect the caltech dataset
    # print("Start Detecting")
    # print(IMG_PATH)
    # write_caltech_results_file(net)                 