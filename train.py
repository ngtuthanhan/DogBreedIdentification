import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import CustomDataset
import scipy.io
from sklearn.model_selection import train_test_split
import os

# Function to save the checkpoint
def save_checkpoint(epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pth')

# Function to load the checkpoint
def load_checkpoint():
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

# Training loop
def train(num_epochs, resume=False):
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print("---------------------------")
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i+1) % 1 == 0:
                train_loss /= (i+1) / batch_size
                train_acc = 100.0 * correct / total
                print(f"Step = {i+1}, Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total
        
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                if (i+1) % 10 == 0:
                    val_loss /= (i+1) / batch_size
                    val_acc = 100.0 * correct / total
                    print(f"Step = {i+1}, Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
                
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        save_checkpoint(epoch + 1)  # Save checkpoint after each epoch

# Reduce model parameter size by 80% using model pruning
def reduce_model_size(amount):
    # Apply model pruning technique (e.g., magnitude-based pruning)
    # Here, we use torch.nn.utils.prune module to prune the model
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=amount
    )

# Retrain the model with reduced parameter size
def retrain(num_epochs):
    # Load the reduced model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    # for epoch in range(num_epochs):

        
if __name__ == '__main__':
    batch_size = 64
    num_workers = 4
    train_split = scipy.io.loadmat('dataset/train_list.mat')
    data_path = 'dataset/Images'
    class_names = os.listdir(data_path)
    num_classes = len(class_names)
    train_files = [os.path.join(data_path, sample[0][0]) for sample in train_split['file_list']]
    train_labels = [file[15:].split('/')[0] for file in train_files] 

    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size = 0.2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformation for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets
    train_dataset = CustomDataset(train_files, train_labels, class_names, transform=transform)
    val_dataset = CustomDataset(val_files, val_labels, class_names, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes (dog breeds)
    model.fc = nn.Linear(2048, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(num_epochs=10, resume=False)  # Train the model for 10 epochs
    
    # Print the size of the model before prunning
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {model_size} parameters")
    reduce_model_size(amount = 0.8)  # Reduce model parameter size by 80%

    # Print the size of the model after reducing model size by 80%
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {model_size} parameters")

    # retrain(num_epochs=5)  # Retrain the model with reduced parameter size