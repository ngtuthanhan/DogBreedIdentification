import torch
import torch.nn.utils.prune
import numpy as np

def save_checkpoint(model, optimizer, checkpoint, epoch):
    """
    Save the model checkpoint along with the optimizer state and current epoch.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        checkpoint (str): Path to save the checkpoint.
        epoch (int): Current epoch.

    """
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint_dict, checkpoint)

def load_checkpoint(model, optimizer, checkpoint, device):
    """
    Load the model checkpoint and optimizer state from the saved checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint (str): Path to the checkpoint file.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int]: The loaded model, optimizer, and the epoch from which to resume training.

    """
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def reduce_model_size(model, amount):
    """
    Reduce the size of the model by applying pruning techniques to its parameters.
    Here, we use torch.nn.utils.prune module to prune the model

    Args:
        model (torch.nn.Module): The model to be pruned.
        amount (float): The amount of pruning to be applied.

    Returns:
        torch.nn.Module: The pruned model.

    """

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model

def get_model_size(model):
    """
    Calculate the total number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to calculate the size.

    Returns:
        int: Total number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

