import torch
import argparse
import os
import scipy.io
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from dataset import handle_data
from model import load_checkpoint, reduce_model_size

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained model')

    # Add the required arguments
    parser.add_argument('--checkpoint_path', type=str, help='Path to the trained model file')
    parser.add_argument('--model', type=str, default='resnet50', help='Model for evaluation')
    parser.add_argument('--data_path', type=str, help='Path to the evaluation data file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for evaluation')
    parser.add_argument('--test_split', type=str, default='dataset/test_split.mat', help='Path to the test split files')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for evaluation')
    parser.add_argument('--reduced_amount', type=float, default=0.0, help='Amount reduction for evaluation')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define required variables
    test_split = scipy.io.loadmat(args.test_split)
    class_names = os.listdir(args.data_path)
    num_classes = len(class_names)
    test_files = [os.path.join(args.data_path, sample[0][0]) for sample in test_split['file_list']]
    test_labels = [file[len(args.data_path)+1:].split('/')[0] for file in test_files] 
    
    # Set the device
    device = torch.device(args.device)

    # Create data loaders
    test_loader = handle_data(test_files, test_labels, class_names, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load pre-trained model
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif args.model == 'resnet152':
        model = models.resnet152(pretrained=False)

    # Modify the last fully connected layer to match the number of classes (dog breeds)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Load model
    model, optimizer, _ = load_checkpoint(model, optimizer, args.checkpoint_path, device)
    model = reduce_model_size(model, amount = args.reduced_amount)
    
    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, criterion, test_loader, device)

    # Print the evaluation results
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")