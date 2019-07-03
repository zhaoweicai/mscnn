
echo "Downloading CityPersons model..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/cascade_mscnn_citypersons_pretrained.zip

echo "Unzipping..."

unzip cascade_mscnn_citypersons_pretrained.zip && rm -f cascade_mscnn_citypersons_pretrained.zip

echo "Done."
