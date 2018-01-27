
echo "Downloading WiderFace model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/mscnn_widerface_pretrained.zip

echo "Unzipping..."

unzip mscnn_widerface_pretrained.zip && rm -f mscnn_widerface_pretrained.zip

echo "Done."
