
echo "Downloading KITTI ped/cyc model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/mscnn_kitti_ped_cyc_pretrained.zip

echo "Unzipping..."

unzip mscnn_kitti_ped_cyc_pretrained.zip && rm -f mscnn_kitti_ped_cyc_pretrained.zip

echo "Done."
