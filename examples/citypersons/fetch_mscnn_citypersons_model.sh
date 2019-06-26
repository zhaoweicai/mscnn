
echo "Downloading CityPersons model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/mscnn_citypersons_pretrained.zip

echo "Unzipping..."

unzip mscnn_citypersons_pretrained.zip && rm -f mscnn_citypersons_pretrained.zip

echo "Done."
