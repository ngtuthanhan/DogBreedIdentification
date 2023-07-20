# wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar -O dataset/annotation.tar
# wget http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat -O dataset/train_data.mat
# wget http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat -O dataset/test_data.mat
# tar -xf dataset/annotation.tar -C dataset

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O dataset/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/list.tar -O dataset/list.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt -O dataset/README.txt
tar -xf dataset/images.tar -C dataset
tar -xf dataset/list.tar -C dataset