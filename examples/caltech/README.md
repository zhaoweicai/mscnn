1. Setup Caltech pedestrian dataset and Piotr Dollar's toolbox by yourself. For the training images, we use 7x set of the original training images, which means every one in four frames is extracted from the continuous video sequences. Make sure you setup the dataset correctly. 

2. For the pretrained model, set `do_bb_norm = 0` in the MATLAB test script, because the pretrained model doesn't use the bounding box de-normalization. For the other models, set `do_bb_norm = 1`.
