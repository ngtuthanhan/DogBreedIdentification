from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):

    def __init__(self, image_paths, labels, class_names, transform=None):
        """
        CustomDataset is a PyTorch Dataset class that represents a custom dataset.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels for each image.
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
