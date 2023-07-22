import streamlit as st
import os
import wget
import tarfile
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
from sklearn.model_selection import train_test_split
from DogBreedIdentification.dataloader.dataset import handle_data
import logging
import pickle
from train import train as train_model
from evaluate import evaluate
from model.model import load_checkpoint
from model.plot import plot_convergence_with_logging_file

def download_files():
    os.makedirs('dataset', exist_ok=True)
    wget.download('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', 'dataset/images.tar')
    wget.download('http://vision.stanford.edu/aditya86/ImageNetDogs/list.tar', 'dataset/list.tar')
    wget.download('http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt', 'dataset/README.txt')
    with tarfile.open('dataset/images.tar', 'r') as tar:
        tar.extractall('dataset')
    with tarfile.open('dataset/list.tar', 'r') as tar:
        tar.extractall('dataset')


def display_training_log(log_file):
    with open(log_file, 'r') as f:
        log_content = f.read()
    st.text(log_content)

# Main Streamlit app
def main():
    st.title("Dog Breed Classification")

    # Dataset path input
    dataset_path = st.text_input("Enter the path to the dataset folder", "dataset/Images")
    st.markdown("If dataset is not provided, then download here")
 
    download_dataset = st.button("Download Dataset", key="download_button", help="Click to download the dataset")


    if download_dataset:
        st.text("Downloading the Stanford dataset, this will take a while...")
        download_files()
        st.success("Dataset downloaded successfully.")

    # Train/test split file input
    train_split_file = st.text_input("Enter the path to the train split file", "dataset/train_list.mat")
    test_split_file = st.text_input("Enter the path to the test split file", "dataset/test_list.mat")
    class_names = os.listdir(dataset_path)
    num_classes = len(class_names)
    train_split = scipy.io.loadmat(train_split_file)
    test_split = scipy.io.loadmat(test_split_file)
    train_files = [os.path.join(dataset_path, sample[0][0]) for sample in train_split['file_list']]
    test_files = [os.path.join(dataset_path, sample[0][0]) for sample in test_split['file_list']]
    train_labels = [file[len(dataset_path)+1:].split('/')[0] for file in train_files] 
    test_labels = [file[len(dataset_path)+1:].split('/')[0] for file in test_files] 

    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size = 0.2)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Checkpoint path input
    checkpoint_path = st.text_input("Enter the path to the checkpoint file", "checkpoints/resnet18.pth")

    # Create data loaders
    train_loader = handle_data(train_files, train_labels, class_names, batch_size=64, num_workers=4)
    val_loader = handle_data(val_files, val_labels, class_names, batch_size=64, num_workers=4)
    test_loader = handle_data(test_files, test_labels, class_names, batch_size=64, num_workers=4)

    # Train or not input
    st.write("You can chose between training model or show the progress of existed trained model")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.write("")
        st.write("")
        st.write("")
        train = st.checkbox("Train Model")
    with col1:
        model_type = st.selectbox("Select the pretrained model", ["resnet18", "resnet34", "resnet50", "resnet152"])
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    
    # Modify the last fully connected layer to match the number of classes (dog breeds)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if train:
        logging_path = st.text_input("Enter the path for saving logging file", "checkpoints/resnet18.log")
        logging.basicConfig(filename=logging_path, filemode='w', level=logging.INFO)
        st.write("Model will be trained for 10 epochs")
        train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, checkpoint_path, logging, num_epochs=1, resume=False, freq_logging=10, streamlit=True)
        test_loss, test_accuracy = evaluate(model, criterion, test_loader, device)
        st.write(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
        convergence_plot = plot_convergence_with_logging_file(logging_path)
        st.pyplot(convergence_plot)
    else:
        logging_path = st.text_input("Enter the path to the training log file", "checkpoints/resnet18.log")
        
        if logging_path:
            display_training_log(logging_path)
            model, optimizer, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
            test_loss, test_accuracy = evaluate(model, criterion, test_loader, device)
            st.write(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
            convergence_plot = plot_convergence_with_logging_file(logging_path)
            st.pyplot(convergence_plot)    

if __name__ == "__main__":
    with open('style/style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    main()
