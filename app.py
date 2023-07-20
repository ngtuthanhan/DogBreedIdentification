import streamlit as st
import os
import wget
import tarfile

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
    st.title("Image Classification")

    # Dataset path input
    dataset_path = st.text_input("Enter the path to the dataset folder", "dataset")
    if not os.path.exists(dataset_path):
        download_dataset = st.button("Download Dataset")
        if download_dataset:
            download_files()

    # Train/test split file input
    split_file = st.text_input("Enter the path to the train/test split file")

    # Checkpoint path input
    checkpoint_path = st.text_input("Enter the path to the checkpoint file")

    # Train or not input
    train = st.checkbox("Train Model")

    if train:
        # Pretrained model selection
        model_type = st.selectbox("Select the pretrained model", ["resnet18", "resnet34", "resnet50", "resnet152"])

        # Training progress display
        # show_training_progress()
        # train_model(dataset_path, split_file, checkpoint_path, model_type)
    else:
        # Training log file input
        log_file = st.text_input("Enter the path to the training log file")

        if log_file:
            display_training_log(log_file)

if __name__ == "__main__":
    main()
