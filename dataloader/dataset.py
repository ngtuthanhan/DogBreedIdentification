from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    """
    CustomDataset is a PyTorch Dataset class that represents a custom dataset.
    """

    def __init__(self, image_paths, labels, class_names, transform=None):
        """
        Initialize the CustomDataset.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels for each image.
            class_names (list): List of class names.
            transform (callable, optional): Optional transform to be applied on each image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]
        label_encode = self.class_names.index(label)

        image = self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label_encode

    def load_image(self, path):
        """
        Loads an image from the given file path.

        Args:
            path (str): File path to the image.

        Returns:
            PIL.Image: The loaded image.
        """
        with Image.open(path) as img:
            return img.convert('RGB')


def handle_data(files, labels, class_names, batch_size, num_workers):
    """
    Handles the data by creating a custom dataset and a data loader.

    Args:
        files (list): List of file paths for the dataset.
        labels (list): List of corresponding labels for each file.
        class_names (list): List of class names.
        batch_size (int): The batch size for the data loader.
        num_workers (int): Number of worker threads for the data loader.

    Returns:
        torch.utils.data.DataLoader: The data loader.

    """
    # Define transformation for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets
    dataset = CustomDataset(files, labels, class_names, transform=transform)

    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader