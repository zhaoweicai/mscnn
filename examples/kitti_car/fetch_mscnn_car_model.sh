
echo "Downloading KITTI car model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/mscnn_kitti_car_pretrained.zip

echo "Unzipping..."

unzip mscnn_kitti_car_pretrained.zip && rm -f mscnn_kitti_car_pretrained.zip

echo "Done."
