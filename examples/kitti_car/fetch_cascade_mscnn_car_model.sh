
echo "Downloading KITTI car model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/cascade_mscnn_kitti_car_pretrained.zip

echo "Unzipping..."

unzip cascade_mscnn_kitti_car_pretrained.zip && rm -f cascade_mscnn_kitti_car_pretrained.zip

echo "Done."
