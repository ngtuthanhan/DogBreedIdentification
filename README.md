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
2. Download and extract the Stanford Dogs dataset
```
mkdir dataset
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O dataset/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar -O dataset/lists.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt -O dataset/README.txt
tar -xf dataset/images.tar -C dataset
tar -xf dataset/lists.tar -C dataset
```
3. Install all required' libraries:
```
pip install -r requirements.txt
```
4. Preprocess data and train the model, for trainning setting, we use SGD for optimizer combine with CosineAnnealling 
```
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet18 --save_path checkpoints/resnet18.pth --logging_path checkpoints/resnet18.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet34 --save_path checkpoints/resnet34.pth --logging_path checkpoints/resnet34.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet50 --save_path checkpoints/resnet50.pth --logging_path checkpoints/resnet50.log
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet152 --save_path checkpoints/resnet152.pth --logging_path checkpoints/resnet152.log
```
We have provided a checkpoint and log files already,
| Model | Checkpoint | Logging file | 
|---|---|---|
| Resnet-18 | [resnet18.pth](https://) | [resnet18.log](https://) | 
| Resnet-34 | [resnet34.pth](https://) | [resnet34.log](https://) | 
| Resnet-50 | [resnet50.pth](https://) | [resnet50.log](https://) | 
| Resnet-152 | [resnet152.pth](https://) | [resnet152.log](https://) |

5. Evaluate the model's performance, for example
```
python3 evaluate.py --data_path dataset/Images --test_split dataset/test_list.mat --model resnet50 --checkpoint_path checkpoints/resnet50.pth
```
The results can be found here, these models were trained same setting, only Resnet-152 was use batch size 32, otherwise 64. We can not deal with the overfitting phenomenon right now, we don't have much time for data augmentation.
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
7. For reducing model's parameters by 80%, we use l1prunning. Network was trained for a short time, then prunes the network and resets it back to initialization. To produce easily, we use the trained network saved in model checkpoint files  `resnet18.pth`,`resnet34.pth`, `resnet50.pth`, `resnet152.pth`.
```
python3 train.py --data_path dataset/Images --train_split dataset/train_list.mat --test_split dataset/test_list.mat --model resnet50 --save_path checkpoints/resnet50_reduction.pth --trained_model_path checkpoints/resnet50.pth --logging_path checkpoints/resnet50_reduction.log --reduced_amount 0.8 
```

We have provided a checkpoint and log files already,
| Model | Checkpoint | Logging file | 
|---|---|---|
| Resnet-18 (reduced model size by 80%) | [resnet18.pth](https://) | [resnet18.log](https://) | 
| Resnet-34 (reduced model size by 80%) | [resnet34.pth](https://) | [resnet34.log](https://) | 
| Resnet-50 (reduced model size by 80%) | [resnet50.pth](https://) | [resnet50.log](https://) | 
| Resnet-152 (reduced model size by 80%) | [resnet152.pth](https://) | [resnet152.log](https://) | 
The results can be found here,
| Model | Test Accuracy |
|---|---|
| Resnet-18 (reduced model size by 80%) | 68.22% |
| Resnet-34 (reduced model size by 80%) | 74.31% |
| Resnet-50 (reduced model size by 80%) |  | 
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