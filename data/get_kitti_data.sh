
echo "Downloading KITTI data..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/kitti_data.zip

echo "Unzipping..."

unzip kitti_data.zip && rm -f kitti_data.zip

echo "Done."
