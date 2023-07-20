import matplotlib.pyplot as plt

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