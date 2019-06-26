
echo "Downloading CityPersons data..."

wget -c http://www.svcl.ucsd.edu/projects/mscnn/citypersons_data.zip

echo "Unzipping..."

unzip citypersons_data.zip && rm -f citypersons_data.zip

echo "Done."
