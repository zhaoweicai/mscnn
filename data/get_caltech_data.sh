
echo "Downloading Caltech data..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/caltech_data.zip

echo "Unzipping..."

unzip caltech_data.zip && rm -f caltech_data.zip

echo "Done."
