import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataset import handle_data
from sklearn.model_selection import train_test_split
import scipy.io
import os
import argparse
import logging
import pickle
import streamlit as st
from model import load_checkpoint, save_checkpoint
from evaluate import evaluate


# Training loop
def train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, checkpoint, logging, num_epochs, resume=False, freq_logging=10, streamlit=False):
    start_epoch = 0

    history = {'train_loss_history': [], 'train_acc_history': [],  'val_loss_history': [], 'val_acc_history': []} 

    if resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint, device)
        with open(file_name + '.pickle', 'rb') as handle:
            history = pickle.load(handle)
    
    for epoch in range(start_epoch, num_epochs):
        if streamlit:
            st.text(f"Epoch: {epoch+1}/{num_epochs}")
            st.text("---------------------------")
        logging.info(f"Epoch: {epoch+1}/{num_epochs}")
        logging.info("---------------------------")
        
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
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i+1) % freq_logging == 0:
                train_loss = train_loss / (i+1) / len(labels)
                train_acc = 100.0 * correct / total
                if streamlit:
                    st.text(f"Step = {i+1}, Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
                logging.info(f"Step = {i+1}, Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total
        if streamlit:
            st.text(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        logging.info(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
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
                if (i+1) % freq_logging == 0:
                    val_loss = val_loss / (i+1) / len(labels)
                    val_acc = 100.0 * correct / total
                    if streamlit:
                        st.write(f"Step = {i+1}, Val Loss: {train_loss:.4f} | Val Accuracy: {train_acc:.2f}%")
                    logging.info(f"Step = {i+1}, Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
                
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        if streamlit:
            st.write(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Save history
        history['train_loss_history'].append(train_loss)
        history['train_acc_history'].append(train_acc)
        history['val_loss_history'].append(val_loss)
        history['val_acc_history'].append(val_acc)
        
    
        save_checkpoint(model, optimizer, checkpoint, epoch + 1)  # Save checkpoint after each epoch

        logger = logging.getLogger()
        file_name = logger.handlers[0].baseFilename
        with open(file_name + '.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    # Add the required arguments
    parser.add_argument('--data_path', type=str, help='Path to the training data file')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to save checkpoints')
    parser.add_argument('--model', type=str, default='resnet50', help='Batch size for training')
    parser.add_argument('--logging_path', type=str, default='training.log', help='Path to save logging file')
    parser.add_argument('--train_split', type=str, default='dataset/train_list.mat', help='Path to the train split files')
    parser.add_argument('--test_split', type=str, default='dataset/test_list.mat', help='Path to the train split files')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for training')
    parser.add_argument('--resume', type=bool, default=False, help='Continue training pre-trained model or not')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Set up logging
    if args.resume:
        logging.basicConfig(filename=args.logging_path, level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=args.logging_path, filemode='w', level=logging.INFO)

    # Define required variables
    train_split = scipy.io.loadmat(args.train_split)
    test_split = scipy.io.loadmat(args.test_split)
    class_names = os.listdir(args.data_path)
    num_classes = len(class_names)
    train_files = [os.path.join(args.data_path, sample[0][0]) for sample in train_split['file_list']]
    test_files = [os.path.join(args.data_path, sample[0][0]) for sample in test_split['file_list']]
    train_labels = [file[len(args.data_path)+1:].split('/')[0] for file in train_files] 
    test_labels = [file[len(args.data_path)+1:].split('/')[0] for file in test_files] 

    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size = 0.2)
    
    # Set the device
    device = torch.device(args.device)

    # Create data loaders
    train_loader = handle_data(train_files, train_labels, class_names, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = handle_data(val_files, val_labels, class_names, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = handle_data(test_files, test_labels, class_names, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load pre-trained model
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model == 'resnet152':
        model = models.resnet152(pretrained=True)

    # Modify the last fully connected layer to match the number of classes (dog breeds)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)


    # Train model
    train(model, optimizer, criterion, scheduler, train_loader, val_loader, device, args.checkpoint_path, logging, num_epochs=args.num_epochs, resume=args.resume)  # Train the model
    
    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, criterion, test_loader, device)

    # Print the evaluation results
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")