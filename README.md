# Dog Breed Classification

## Introduction
This project leverages the ResNet (Residual Neural Network) architecture, a deep learning model known for its excellent performance on image recognition tasks. Additionally, the model is pruned to reduce its size while preserving its accuracy. 

## Dataset
To train and evaluate the dog breed classification model, we will use the Stanford Dogs dataset. You can download the dataset from the following link: [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)

The dataset contains 20580 labeled images of different dog breeds, allowing us to train our model effectively. It covers a wide variety of breeds, ensuring the model's ability to generalize well to unseen examples.

## Model Training
To train the dog breed classification model, follow these steps:
1. Clone this GitHub repository and navigate to the project's directory.
```
git clone https://github.com/ngtuthanhan/DogBreedIdentification.git
cd DogBreedIdentification
```
2. Download and extract the Stanford Dogs dataset
```
mkdir dataset
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O dataset/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar -O dataset/lists.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt -O dataset/README.txt
tar -xf dataset/images.tar -C dataset
tar -xf dataset/lists.tar -C dataset
```
3. Install all required libraries:
```
pip install -r requirements.txt
```
4. Preprocess data and train the model, for training setting, we use SGD for optimizer combined with CosineAnnealling 
```
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet18 --save_path checkpoints/resnet18.pth --logging_path checkpoints/resnet18.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet34 --save_path checkpoints/resnet34.pth --logging_path checkpoints/resnet34.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet50 --save_path checkpoints/resnet50.pth --logging_path checkpoints/resnet50.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet152 --save_path checkpoints/resnet152.pth --logging_path checkpoints/resnet152.log
```
We have provided a checkpoint and log files already,
| Model | Checkpoint | Logging file | 
|---|---|---|
| Resnet-18 (reduced model size by 80%) | [resnet18.pth](https://drive.google.com/file/d/1pFR-LxpWXWZanBMuBlr_GGy3nYjEUn0B/) | [resnet18.log](https://drive.google.com/file/d/1OWyozYko3a3Z4ExK03dwLLEkAQci0O1U/) | 
| Resnet-34 (reduced model size by 80%) | [resnet34.pth](https://drive.google.com/file/d/1MF9Dd0lo5lUNsbc43olkyO_U1Xr0pTXR/) | [resnet34.log](https://drive.google.com/file/d/1vjF16NIHQZxxdpAzJG72KPVHhEuR-T-d/) | 
| Resnet-50 (reduced model size by 80%) | [resnet50.pth](https://drive.google.com/file/d/1K6203r2D-5O_-C885ToP15nueHtUBImf/) | [resnet50.log](https://drive.google.com/file/d/1m9WEEYx9pTcYiVgqo7-9XXvHs500GEaV/) | 
| Resnet-152 (reduced model size by 80%) | [resnet152.pth](https://drive.google.com/file/d/1eQ7LEsGi8xyFlypFM9KIDAWTvWhVtNaX/) | [resnet152.log](https://drive.google.com/file/d/1ekNi5ZCrW4BdwRya93NgsDxADkZANFKP/) | 

5. Evaluate the model's performance, for example
```
python3 evaluate.py --data_path dataset/Images --test_split dataset/test_list.mat --model resnet50 --checkpoint_path checkpoints/resnet50.pth
```
The results can be found here, these models were trained in the same setting, only Resnet-152 used batch size 32, otherwise 64. We can not deal with the overfitting phenomenon right now, we don't have much time for data augmentation.
| Model | Test Accuracy | 
|---|---|
| Resnet-18 | 74.20% | 
| Resnet-34 | 78.53% | 
| Resnet-50 | 81.68% | 
| Resnet-152 | 79.08% | 
6. Resume training the model, for example
```
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet18 --checkpoint_path checkpoints/resnet18.pth --logging_path checkpoints/resnet18.log --resume True --num_epochs 200
```
7. For reducing the model's parameters by 80%, we use l1prunning. The network was trained for a short time, then prunes the network and resets it back to initialization. To produce easily, we use the trained network saved in model checkpoint files  `resnet18.pth`, `resnet34.pth`, `resnet50.pth`, `resnet152.pth`.
```
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet50 --save_path checkpoints/resnet50_reduction.pth --trained_model_path checkpoints/resnet50.pth --logging_path checkpoints/resnet50_reduction.log --reduced_amount 0.8 
```

We have provided a checkpoint and log files already,
| Model | Checkpoint | Logging file | 
|---|---|---|
| Resnet-18 (reduced model size by 80%) | [resnet18.pth](https://drive.google.com/file/d/10mu9hmTkj7Igl7H1o4ZGJ3MdwPDGhLNi/) | [resnet18.log](https://drive.google.com/file/d/10ybKlGZ_JxWaoPR9Xg9faWeBQTG7OTs6/) | 
| Resnet-34 (reduced model size by 80%) | [resnet34.pth](https://drive.google.com/file/d/1ahAfxluCZ0-aqSUYFC3sby278U4BgeMR/) | [resnet34.log](https://drive.google.com/file/d/1GR7Tma0dNtX-NuTLzfivQm0zGTmp0xOk/) | 
| Resnet-50 (reduced model size by 80%) | [resnet50.pth](https://drive.google.com/file/d/1c2hqvX_nHSvFd52R6_RI5n4fH60LnEsY/) | [resnet50.log](https://drive.google.com/file/d/1SJdZ1E0JCFeA5G3KLeCoSPTbmfHcp509/) | 
| Resnet-152 (reduced model size by 80%) | [resnet152.pth](https://drive.google.com/file/d/1A4nOLcHSRj16y9BikFR9sxHogAePnNoz/) | [resnet152.log](https://drive.google.com/file/d/1XQ2SDBFdyXZTzmopmqS3n2T0YK0qHAbF/) | 

8. Evaluate the performance of the model which was reduced model size, for example
```
python3 evaluate.py --data_path dataset/Images --test_split dataset/test_list.mat --model resnet50 --checkpoint_path checkpoints/resnet50.pth --reduced_model True
```
The results can be found here,
| Model | Test Accuracy |
|---|---|
| Resnet-18 (reduced model size by 80%) | 68.22% |
| Resnet-34 (reduced model size by 80%) | 74.31% |
| Resnet-50 (reduced model size by 80%) | 77.83% | 
| Resnet-152 (reduced model size by 80%) | 78.59% |

## Model Deployment
To deploy the dog breed classification model using Streamlit, follow these steps:
1. Run the Streamlit application:
```
streamlit run app.py
```
The application will launch in your browser in port 8501
## License
This project is licensed under the MIT License, granting users the freedom to modify, distribute, and use the code for both personal and commercial purposes.
