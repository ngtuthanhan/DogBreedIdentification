# Dog Breed Classification

## Introduction
This project leverages the ResNet (Residual Neural Network) architecture, a deep learning model known for its excellent performance on image recognition tasks. Additionally, the model is pruned to reduce its size while preserving its accuracy. 

## Dataset
To train and evaluate the dog breed classification model, we will be using the Stanford Dogs dataset. You can download the dataset from the following link: [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)

The dataset contains 20580 labeled images of different dog breeds, allowing us to train our model effectively. It covers a wide variety of breeds, ensuring the model's ability to generalize well to unseen examples.

## Model Training
To train the dog breed classification model, follow these steps:
1. Clone this GitHub repository and navigate to the project's directory.
```
git clone https://github.com/ngtuthanhan/DogBreedIdentification.git
cd dog-breed-classification
```
2. Download and extract the Stanford Dogs dataset, running `bash download_dataset.sh` or excuting
```
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O dataset/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/list.tar -O dataset/list.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt -O dataset/README.txt
tar -xf dataset/images.tar -C dataset
tar -xf dataset/list.tar -C datasetash
```
3. Preprocess data and train the model
4. Evaluate the model's performance
5. Finetune the model


## Model Deployment
To deploy the dog breed classification model using Streamlit, follow these steps:
1. Run the Streamlit application:
```
streamlit run app.py
```
The application will launch in your browser

## License

This project is licensed under the MIT License, granting users the freedom to modify, distribute, and use the code for both personal and commercial purposes.