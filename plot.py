import matplotlib.pyplot as plt
import pickle

def plot_convergence(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    epochs = range(1, len(train_loss_history) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Train')
    plt.plot(epochs, val_loss_history, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Train')
    plt.plot(epochs, val_acc_history, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    plt.show()
    return plt

def plot_convergence_with_logging_file(log_file):
    history = {'train_loss_history': [], 'train_acc_history': [],  'val_loss_history': [], 'val_acc_history': []} 
    with open(log_file, 'r') as f:
        loggings = f.readlines()
    for logging in loggings:
        if 'Step = ' in logging:
            continue
        if 'Train ' in logging:
            train_loss_start_idx = logging.find('Train Loss:') + len('Train Loss: ') 
            train_loss_end_idx = logging.find(' | Train Accuracy: ')
            train_loss = float(logging[train_loss_start_idx:train_loss_end_idx])

            train_acc_start_idx = logging.find('Train Accuracy: ') + len('Train Accuracy: ')
            train_acc_end_idx = logging.find('%')
            train_acc = float(logging[train_acc_start_idx:train_acc_end_idx])

            history['train_loss_history'].append(train_loss)
            history['train_acc_history'].append(train_acc)
        
        if 'Val ' in logging:
            val_loss_start_idx = logging.find('Val Loss:') + len('Val Loss: ') 
            val_loss_end_idx = logging.find(' | Val Accuracy: ')
            val_loss = float(logging[val_loss_start_idx:val_loss_end_idx])
            
            val_acc_start_idx = logging.find('Val Accuracy: ') + len('Val Accuracy: ')
            val_acc_end_idx = logging.find('%')
            val_acc = float(logging[val_acc_start_idx:val_acc_end_idx])

            history['val_loss_history'].append(val_loss)
            history['val_acc_history'].append(val_acc)
    
    return plot_convergence( history['train_loss_history'], history['val_loss_history'], 
                            history['train_acc_history'], history['val_acc_history'])

if __name__ == '__main__':
    log_file = 'checkpoints/resnet18.log'
    plot_convergence_with_logging_file(log_file)
