% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

addpath('../../matlab/');
addpath('../../utils/');

root_dir = 'mscnn-12s-2x-pretrained/';
binary_file = [root_dir 'mscnn_widerface_trainval_2nd_iter_35000.caffemodel'];
assert(exist(binary_file, 'file') ~= 0);
definition_file = [root_dir 'mscnn_deploy.prototxt'];
assert(exist(definition_file, 'file') ~= 0);
use_gpu = true;
if (~use_gpu)
  caffe.set_mode_cpu();
else
  caffe.set_mode_gpu();  
  gpu_id = 0; caffe.set_device(gpu_id);
end
% Initialize a network
net = caffe.Net(definition_file, binary_file, 'test');

% set widerface dataset directory
root_dir = '/your/WiderFace/path/';
image_dir = [root_dir 'test/images/'];
load([root_dir 'wider_face_split/wider_face_test.mat']);
comp_id = 'widerface_12s_2x_35k_test';
numEvent = numel(file_list);

% use the orignal image size
imgW = 0; imgH = 0; max_size = 3072;

mu = ones(1,1,3); mu(:,:,1:3) = [104 117 123];

% bbox de-normalization parameters
do_bb_norm = 1;
if (do_bb_norm)
  bbox_means = [0 0 0 0];
  bbox_stds = [0.1 0.1 0.2 0.2];
else
  bbox_means = [];
  bbox_stds = [];
end

% non-maxisum suppression parameters
pNms.type = 'maxg'; pNms.overlap = 0.3; pNms.ovrDnm = 'union';

cls_ids = [2]; num_cls=length(cls_ids); 
obj_names = {'bg','face'};
final_detect_boxes = cell(numEvent,1); 
final_proposals = cell(numEvent,1);
proposal_thr = -5; usedtime=0; count_id = 1;

show = 1; show_thr = 0.3; det_thr = 0.05;
if (show)
  fig = figure(1); set(fig,'Position',[-30 30 800 800]);
  hh.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

for k = 1:numEvent
  eventImgs = file_list{k}; event_name = event_list{k};
  numImg = numel(eventImgs);
  event_detect_boxes = cell(numImg,1); 
  event_proposals = cell(numImg,1);
  for ii = 1:numImg
      img_path = [image_dir event_name '/' eventImgs{ii} '.jpg'];
      test_image = imread(img_path); 
      if (show), imshow(test_image); end
      [orgH,orgW,~] = size(test_image);
      if (imgW==0), rzW = orgW; else rzW = imgW; end;
      if (imgH==0), rzH = orgH; else rzH = imgH; end;
      rzW = round(rzW/32)*32;
      rzH = round(rzH/32)*32;
      if (rzH>max_size || rzW>max_size)
        tmpratio = max_size/max(rzH,rzW);
        rzH = round(rzH*tmpratio/32)*32; 
        rzW = round(rzW*tmpratio/32)*32; 
      end
      ratios = [rzH rzW]./[orgH orgW];
      test_image = imresize(test_image,[rzH rzW]); 
      rzmu = repmat(mu,[rzH,rzW,1]);
      test_image = single(test_image(:,:,[3 2 1]));
      test_image = bsxfun(@minus,test_image,rzmu);
      test_image = permute(test_image, [2 1 3]);

      % network forward
      tic; outputs = net.forward({test_image}); pertime=toc;
      usedtime=usedtime+pertime; avgtime=usedtime/count_id;

      tmp=squeeze(outputs{1}); bbox_preds = tmp';
      tmp=squeeze(outputs{2}); cls_pred = tmp'; 
      tmp=squeeze(outputs{3}); tmp = tmp'; tmp = tmp(:,2:end); 
      tmp(:,3) = tmp(:,3)-tmp(:,1); tmp(:,4) = tmp(:,4)-tmp(:,2); 
      proposal_pred = tmp; proposal_score = proposal_pred(:,end);

      % filtering some bad proposals
      keep_id = find(proposal_score>=proposal_thr & proposal_pred(:,3)~=0 & proposal_pred(:,4)~=0);
      proposal_pred = proposal_pred(keep_id,:); 
      bbox_preds = bbox_preds(keep_id,:); cls_pred = cls_pred(keep_id,:);

      proposals = double(proposal_pred);
      proposals(:,1) = proposals(:,1)./ratios(2); 
      proposals(:,3) = proposals(:,3)./ratios(2);
      proposals(:,2) = proposals(:,2)./ratios(1);
      proposals(:,4) = proposals(:,4)./ratios(1);
      event_proposals{ii} = proposals;

      for i = 1:num_cls
        id = cls_ids(i); bbset = [];
        bbox_pred = bbox_preds(:,id*4-3:id*4); 
        proposals = event_proposals{ii};

        % bbox de-normalization
        if (do_bb_norm)
          bbox_pred = bbox_pred.*repmat(bbox_stds,[size(bbox_pred,1) 1]);
          bbox_pred = bbox_pred+repmat(bbox_means,[size(bbox_pred,1) 1]);
        end

        exp_score = exp(cls_pred);
        sum_exp_score = sum(exp_score,2);
        prob = exp_score(:,id)./sum_exp_score; 
        ctr_x = proposal_pred(:,1)+0.5*proposal_pred(:,3);
        ctr_y = proposal_pred(:,2)+0.5*proposal_pred(:,4);
        tx = bbox_pred(:,1).*proposal_pred(:,3)+ctr_x;
        ty = bbox_pred(:,2).*proposal_pred(:,4)+ctr_y;
        tw = proposal_pred(:,3).*exp(bbox_pred(:,3));
        th = proposal_pred(:,4).*exp(bbox_pred(:,4));
        tx = tx-tw/2; ty = ty-th/2;
        tx = tx./ratios(2); tw = tw./ratios(2);
        ty = ty./ratios(1); th = th./ratios(1);

        % clipping bbs to image boarders
        tx = max(0,tx); ty = max(0,ty);
        tw = min(tw,orgW-tx); th = min(th,orgH-ty);     
        bbset = double([tx ty tw th prob]);
        if (det_thr>0)
          keep_id = find(prob>=det_thr);
          bbset = bbset(keep_id,:);
          proposals = proposals(keep_id,:);
        end
        idlist = 1:size(bbset,1); bbset = [bbset idlist'];
        bbset=bbNms(bbset,pNms);
        event_detect_boxes{ii,i} = bbset(:,1:5);

        if (show) 
          proposals_show = zeros(0,5); bbs_show = zeros(0,6);
          if (size(bbset,1)>0) 
            show_id = find(bbset(:,5)>=show_thr);
            bbs_show = bbset(show_id,:);
            proposals_show = proposals(bbs_show(:,6),:); 
          end
          % detection
          for j = 1:size(bbs_show,1)
            rectangle('Position',bbs_show(j,1:4),'EdgeColor','y','LineWidth',2);
            show_text = sprintf('%.2f',bbs_show(j,5));
            x = bbs_show(j,1)+0.5*bbs_show(j,3);
            text(x,bbs_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
                'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
          end  
        end
      end
      count_id = count_id+1;
  end
  final_detect_boxes{k} = event_detect_boxes; 
  final_proposals{k} = event_proposals;
  fprintf('event %i/%i, avgtime=%.4fs\n',k,numEvent,avgtime); 
end

% save results for evaluation
comp_id_dir = ['./detections/' comp_id '/'];
if (~exist(comp_id_dir))
  mkdir(comp_id_dir);
end
for k = 1:numEvent
  eventImgs = file_list{k}; event_name = event_list{k};
  event_dir = [comp_id_dir event_name '/'];
  event_detect_boxes = final_detect_boxes{k,1};
  if (~exist(event_dir))
    mkdir(event_dir);
  end
  for i=1:numel(eventImgs)
    result_img_path = [event_dir eventImgs{i} '.txt'];
    fid = fopen(result_img_path, 'wt');
    fprintf(fid, '%s\n', eventImgs{i});
    dets = event_detect_boxes{i};
    numDets = size(dets,1);
    fprintf(fid, '%d\n', numDets);
    for j=1:numDets
      fprintf(fid, '%d %d %d %d %f\n', round(dets(j,1)), round(dets(j,2)),...
          round(dets(j,3)), round(dets(j,4)), dets(j,5));
    end
    fclose(fid);
  end
end

caffe.reset_all();
