% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

addpath('../../matlab/');
addpath('../../utils/');

root_dir = 'cascade-mscnn-12s-align-pretrained/';
binary_file = [root_dir 'mscnn_widerface_train_2nd_iter_25000.caffemodel'];
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
data_dir = '/your/WiderFace/path/';
image_dir = [data_dir 'val/images/'];
load([data_dir 'wider_face_split/wider_face_val.mat']);
comp_id = 'cascade_mscnn_12s_align_pretrained';
numEvent = numel(file_list);

% architecture
if(~isempty(strfind(root_dir, 'cascade'))), CASCADE = 1;
else CASCADE = 0; end
if (~CASCADE)
  % baseline model
  proposal_blob_names = {'proposals'};
  bbox_blob_names = {'output_bbox_1st'};
  cls_prob_blob_names = {'cls_prob_1st'};
  output_names = {'1st'};
else
  % cascade-rcnn model
  proposal_blob_names = {'proposals_3rd'};
  bbox_blob_names = {'output_bbox_3rd'};
  cls_prob_blob_names = {'cls_prob_3rd_avg'};
  output_names = {'3rd_avg'};
end
num_outputs = numel(proposal_blob_names);
assert(num_outputs==numel(bbox_blob_names));
assert(num_outputs==numel(cls_prob_blob_names));
assert(num_outputs==numel(output_names));

% use the orignal image size
imgW = 0; imgH = 0; max_size = 3072;

mu = ones(1,1,3); mu(:,:,1:3) = [104 117 123];

% non-maxisum suppression parameters
pNms.type = 'maxg'; pNms.overlap = 0.3; pNms.ovrDnm = 'union';

cls_ids = [2]; num_cls=length(cls_ids); 
obj_names = {'bg','face'};
final_detect_boxes = cell(numEvent,1); 
proposal_thr = -5; usedtime=0; count_id = 1;

show = 1; show_thr = 0.3; det_thr = 0.05;
if (show)
  fig = figure(1); set(fig,'Position',[-30 30 800 800]);
  hh.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

for k = 1:numEvent
  eventImgs = file_list{k}; event_name = event_list{k};
  numImg = numel(eventImgs);
  event_detect_boxes = cell(numImg,num_outputs,1); 
  for ii = 1:numImg
      img_path = [image_dir event_name '/' eventImgs{ii} '.jpg'];
      test_image = imread(img_path); 
      org_img = test_image;
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
      
      for nn = 1:num_outputs
        if (show), imshow(org_img); end
        tmp = squeeze(net.blobs(bbox_blob_names{nn}).get_data()); 
        tmp = tmp'; tmp = tmp(:,2:end);
        tmp(:,[1,3]) = tmp(:,[1,3])./ratios(2);
        tmp(:,[2,4]) = tmp(:,[2,4])./ratios(1);
        % clipping bbs to image boarders
        tmp(:,[1,2]) = max(0,tmp(:,[1,2]));
        tmp(:,3) = min(tmp(:,3),orgW); tmp(:,4) = min(tmp(:,4),orgH);
        tmp(:,[3,4]) = tmp(:,[3,4])-tmp(:,[1,2])+1;
        output_bboxs = double(tmp);  

        tmp = squeeze(net.blobs(cls_prob_blob_names{nn}).get_data()); 
        cls_prob = tmp'; 

        tmp = squeeze(net.blobs(proposal_blob_names{nn}).get_data());
        tmp = tmp'; tmp = tmp(:,2:end); 
        tmp(:,[3,4]) = tmp(:,[3,4])-tmp(:,[1,2])+1; 
        proposals = tmp;

        keep_id = find(proposals(:,3)~=0 & proposals(:,4)~=0);
        proposals = proposals(keep_id,:); 
        output_bboxs = output_bboxs(keep_id,:); cls_prob = cls_prob(keep_id,:);
        
        for i = 1:num_cls
          id = cls_ids(i); 
          prob = cls_prob(:,id);  
          bbset = double([output_bboxs prob]);
          if (det_thr>0)
            keep_id = find(prob>=det_thr); bbset = bbset(keep_id,:);
          end
          bbset=bbNms(bbset,pNms);
          event_detect_boxes{ii,nn,i} = bbset(:,1:5);

          if (show) 
            bbs_show = zeros(0,6);
            if (size(bbset,1)>0) 
              show_id = find(bbset(:,5)>=show_thr);
              bbs_show = bbset(show_id,:);
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
      end 
      count_id = count_id+1;
  end
  final_detect_boxes{k} = event_detect_boxes; 
  fprintf('event %i/%i, avgtime=%.4fs\n',k,numEvent,avgtime); 
end

% save results for evaluation
for nn = 1:num_outputs
    comp_id_dir = ['./detections/' comp_id '_' output_names{nn} '/'];
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
        dets = event_detect_boxes{i, nn};
        numDets = size(dets,1);
        fprintf(fid, '%d\n', numDets);
        for j=1:numDets
          fprintf(fid, '%d %d %d %d %f\n', round(dets(j,1)), round(dets(j,2)),...
              round(dets(j,3)), round(dets(j,4)), dets(j,5));
        end
        fclose(fid);
      end
    end
end

caffe.reset_all();
