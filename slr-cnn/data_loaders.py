"""
Loads tensors using PyTorch
"""

import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ImageLabelDataset(Dataset):
    """
    Generic class for reading images and features from a dataset

    Parameters:
    ----------
    images : Torch.tensor
        Images stored in tensor
    labels : Torch.tensor
        Labels stored in tensor
    """

    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_loader(data, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    """
    PyTorch DataLoader with optional shuffling and fixed random seed

    Parameters:
    ----------
    data : torch.utils.data.Dataset
        Dataset containing images and labels
    batch_size : int
        Number of samples per batch to load
    shuffle : bool
        For invoking randomly shuffled data at each epoch (see https://docs.pytorch.org/docs/stable/data.html)

    Returns:
    -------
    torch.utils.data.DataLoader
        DataLoader instance for input dataset
    """
    # Add manual seed to retain same randomly generated numbers
    #g = torch.Generator()
    #g.manual_seed(0)

    return torch.utils.data.DataLoader(
        data,
        batch_size = batch_size,
        shuffle = shuffle
    )