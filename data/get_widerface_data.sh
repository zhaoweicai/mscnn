
echo "Downloading WiderFace data..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/widerface_data.zip

echo "Unzipping..."

unzip widerface_data.zip && rm -f widerface_data.zip

echo "Done."
