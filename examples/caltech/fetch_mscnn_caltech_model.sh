
echo "Downloading Caltech model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/mscnn_caltech_pretrained.zip

echo "Unzipping..."

unzip mscnn_caltech_pretrained.zip && rm -f mscnn_caltech_pretrained.zip

echo "Done."
