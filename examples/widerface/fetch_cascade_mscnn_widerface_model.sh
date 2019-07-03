
echo "Downloading WiderFace model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/cascade_mscnn_widerface_pretrained.zip

echo "Unzipping..."

unzip cascade_mscnn_widerface_pretrained.zip && rm -f cascade_mscnn_widerface_pretrained.zip

echo "Done."
