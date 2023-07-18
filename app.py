import torch
import streamlit as st
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Import your custom dataset and model classes
from dataset import CustomDataset

def train_model(dataset_path, pretrained_model_path, model_type):
    # Load the dataset and create data loader
    # You may need to modify this based on your dataset structure and requirements
    train_dataset = CustomDataset(dataset_path, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Load or initialize the model
    if pretrained_model_path:
        model = torch.load(pretrained_model_path)
    else:
        # Create the model based on the selected model type
        if model_type == "resnet50":
            model = models.resnet50(pretrained=False)
        elif model_type == "resnet152":
            model = models.resnet152(pretrained=False)
        else:
            st.error("Invalid model type selected.")
            return

        # Modify the last fully connected layer for the number of classes in the dataset
        num_classes = len(train_dataset.classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Move the model to device (e.g., GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # You can modify this value
    for epoch in range(num_epochs):
        st.text(f"Epoch: {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        st.text(f"Train Loss: {train_loss:.4f}")

        # Add your code to display accuracy vs loss across epoch (e.g., plot using matplotlib or Streamlit components)

    st.text("Training Completed.")

def main():
    st.title("Dog Breed Classification")
    st.sidebar.title("Configuration")

    # User input fields
    dataset_path = st.sidebar.text_input("Dataset Path", "")
    pretrained_model_path = st.sidebar.text_input("Pretrained Model Path (optional)", "")
    model_type = st.sidebar.selectbox("Model Type", ["resnet50", "resnet152"])

    if st.sidebar.button("Train Model"):
        train_model(dataset_path, pretrained_model_path, model_type)

if __name__ == "__main__":
    main()
