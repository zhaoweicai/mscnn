% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

addpath('../../matlab/');
addpath('../../utils/');

root_dir = 'mscnn-7s-720-pretrained/';
binary_file = [root_dir 'mscnn_caltech_train_2nd_iter_20000.caffemodel'];
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

% set KITTI dataset directory
root_dir = '/your/Caltech/path/';
image_dir = [root_dir 'test/images/'];
comp_id = 'caltech_7s_720_20k_test';
image_list = dir([image_dir '*.jpg']); 
nImg=length(image_list);

% choose the right input size
% imgW = 640; imgH = 480;
imgW = 960; imgH = 720;

mu = ones(1,1,3); mu(:,:,1:3) = [104 117 123];
mu = repmat(mu,[imgH,imgW,1]);

% bbox de-normalization parameters
do_bb_norm = 0; % set 0 only for the pretrained model
if (do_bb_norm)
  bbox_means = [0 0 0 0];
  bbox_stds = [0.1 0.1 0.2 0.2];
else
  bbox_means = [];
  bbox_stds = [];
end

% non-maxisum suppression parameters
pNms.type = 'maxg'; pNms.overlap = 0.5; pNms.ovrDnm = 'union';

cls_ids = [2]; num_cls=length(cls_ids); 
obj_names = {'bg','ped'};
final_detect_boxes = cell(nImg,num_cls); final_proposals = cell(nImg,1);
proposal_thr = -10; usedtime=0; 

show = 1; show_thr = 0.1;
if (show)
  fig=figure(1); set(fig,'Position',[50 100 640 480]);
  h.axes = axes('position',[0,0,1,1]);
end

for k = 1:nImg
  test_image = imread([image_dir image_list(k).name]);
  if (show)
    imshow(test_image,'parent',h.axes); axis(h.axes,'image','off'); hold(h.axes,'on');
  end
  [orgH,orgW,~] = size(test_image);
  ratios = [imgH imgW]./[orgH orgW];
  test_image = imresize(test_image,[imgH imgW]); 
  test_image = single(test_image(:,:,[3 2 1]));
  test_image = bsxfun(@minus,test_image,mu);
  test_image = permute(test_image, [2 1 3]);

  % network forward
  tic; outputs = net.forward({test_image}); pertime=toc;
  usedtime=usedtime+pertime; avgtime=usedtime/k;
    
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
  final_proposals{k} = proposals;

  for i = 1:num_cls
    id = cls_ids(i); bbset = [];
    bbox_pred = bbox_preds(:,id*4-3:id*4); 

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
    idlist = 1:size(bbset,1); bbset = [bbset idlist'];
    bbset=bbNms(bbset,pNms);
    final_detect_boxes{k,i} = bbset(:,1:5);

    if (show) 
      proposals_show = zeros(0,5); bbs_show = zeros(0,6);
      if (size(bbset,1)>0) 
        show_id = find(bbset(:,5)>=show_thr);
        bbs_show = bbset(show_id,:);
        proposals_show = proposals(bbs_show(:,6),:); 
      end
      % proposal
      for j = 1:size(proposals_show,1)
        rectangle('Position',proposals_show(j,1:4),'EdgeColor','g','LineWidth',2);
        show_text = sprintf('%.2f',proposals_show(j,5));
        x = proposals_show(j,1)+0.5*proposals_show(j,3);
        text(x,proposals_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
            'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
      end 
      % detection
      for j = 1:size(bbs_show,1)
        rectangle('Position',bbs_show(j,1:4),'EdgeColor','y','LineWidth',2);
        show_text = sprintf('%s=%.2f',obj_names{id},bbs_show(j,5));
        x = bbs_show(j,1)+0.5*bbs_show(j,3);
        text(x,bbs_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
            'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
      end  
    end
  end
  if (mod(k,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime); end
end

for i=1:nImg
  for j=1:num_cls
    final_detect_boxes{i,j}=[ones(size(final_detect_boxes{i,j},1),1)*i final_detect_boxes{i,j}]; 
  end
  final_proposals{i}=[ones(size(final_proposals{i},1),1)*i final_proposals{i}];
end
for j=1:num_cls
  id = cls_ids(j);
  save_detect_boxes=cell2mat(final_detect_boxes(:,j));
  dlmwrite(['detections/' comp_id '_' obj_names{id} '.txt'],save_detect_boxes);
end
final_proposals=cell2mat(final_proposals);
%dlmwrite(['proposals/' comp_id '.txt'],final_proposals);

caffe.reset_all();

