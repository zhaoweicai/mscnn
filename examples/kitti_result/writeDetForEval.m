% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

% clear and close everything
clear all; close all;
kitti_dir = '/your/KITTI/path/';
addpath([kitti_dir 'devkit_object/devkit/matlab/']);

data_set = 'val'; % or 'test'
if (strcmp(data_set,'test')) 
  is_gt_available = 0;
else
  is_gt_available = 1;
end
gt_dir = [kitti_dir 'data_object_label_2/training/label_2'];

% get the list for evaluation
list_dir = ['../../data/kitti/ImageSets/' data_set '.txt'];
test_id = load(list_dir);
nimages = length(test_id);

% load detection results
car_dets_path = '../kitti_car/detections/kitti_7s_576_2x_25k_val_car.txt';
if (exist(car_dets_path))
  car_dets = load(car_dets_path);
else
  car_dets = zeros(0,6);
end
ped_dets_path = '../kitti_ped_cyc/detections/kitti_7s_576_2x_25k_val_ped.txt';
if (exist(ped_dets_path))
  ped_dets = load(ped_dets_path);
else
  ped_dets = zeros(0,6);
end
cyc_dets_path = '../kitti_ped_cyc/detections/kitti_7s_576_2x_25k_val_cyc.txt';
if (exist(cyc_dets_path))
  cyc_dets = load(cyc_dets_path);
else
  cyc_dets = zeros(0,6);
end

score_scale = 1000;
comp_id = 'vgg_7s_576_2x_25k_val';
result_dir = [data_set '/' comp_id '/'];
save_dir = [result_dir 'data/'];
if (~exist(save_dir)), mkdir(save_dir); end

for i = 1:nimages
    if (mod(i,1000)==0), fprintf('idx: %i / %i\n',i,nimages); end
    objects=[]; num = 0;
    
    % car
    idx = find(car_dets(:,1)==i);
    bbs = car_dets(idx,2:6);
    bbs(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
    for j = 1:size(bbs,1)
      num = num+1;
      objects(num).type = 'Car'; objects(num).score = bbs(j,5)*score_scale;
      objects(num).x1 = bbs(j,1); objects(num).y1 = bbs(j,2);
      objects(num).x2 = bbs(j,3); objects(num).y2 = bbs(j,4);
    end
    
    % pedestrian
    idx = find(ped_dets(:,1)==i);
    bbs = ped_dets(idx,2:6);
    bbs(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
    for j = 1:size(bbs,1)
      num = num+1;
      objects(num).type = 'Pedestrian'; objects(num).score = bbs(j,5)*score_scale;
      objects(num).x1 = bbs(j,1); objects(num).y1 = bbs(j,2);
      objects(num).x2 = bbs(j,3); objects(num).y2 = bbs(j,4);
    end
    
    % cyclist
    idx = find(cyc_dets(:,1)==i);
    bbs = cyc_dets(idx,2:6);
    bbs(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
    for j = 1:size(bbs,1)
      num = num+1;
      objects(num).type = 'Cyclist'; objects(num).score = bbs(j,5)*score_scale;
      objects(num).x1 = bbs(j,1); objects(num).y1 = bbs(j,2);
      objects(num).x2 = bbs(j,3); objects(num).y2 = bbs(j,4);
    end
    
    img_idx = test_id(i);
    writeLabels(objects,save_dir,img_idx);
end

if (is_gt_available)
  plot_dir = [result_dir 'plot/'];
  % input arguments [gt_dir, result_dir, list_dir];
  command_line = sprintf('eval/evaluate_object %s %s %s', gt_dir,result_dir,list_dir);
  system(command_line);
  plot_set = dir([plot_dir '*.txt']);
  for i = 1:length(plot_set)
    results = load([plot_dir plot_set(i).name]);
    x = results(:,1); fig=figure(i);
    h1 = plot(x,results(:,2),'LineWidth',3,'Color','r'); hold on;
    easy_legend = sprintf('%s %%%.02f','Easy',100*mean(results(1:4:41,2)));
    h2 = plot(x,results(:,3),'LineWidth',3,'Color','g'); hold on;
    moderate_legend = sprintf('%s %%%.02f','Moderate',100*mean(results(1:4:41,3)));
    h3 = plot(x,results(:,4),'LineWidth',3,'Color','b'); hold on;
    hard_legend = sprintf('%s %%%.02f','Hard',100*mean(results(1:4:41,4)));
    hd=legend([h1 h2 h3], easy_legend, moderate_legend, hard_legend);
    set(hd,'FontSize',18,'Location','SouthWest'); grid;
    tt=get(gca,'Title'); title(plot_set(i).name(1:end-14)); set(tt,'FontSize',18); 
    saveas(fig,[result_dir plot_set(i).name(1:end-4) '.png'])
  end
end


