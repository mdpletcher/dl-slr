import os


import numpy as np
import torch

from typing import Dict, Callable, Optional
from data_loaders import ImageLabelDataset, create_loader
from dataclasses import dataclass, field

@dataclass
class DataSetup:
    """
    Setup training and validation dataloaders

    Parameters:
    ----------
    batch_size : int
        Number of images in batch
    data_path : str
        Path to .pt tensors containing train/val data
    """

    batch_size: int
    data_path: str

    dataloaders: Dict[str, torch.utils.data.DataLoader] = field(init = False)
    #train_data: 

    def get_loaders(
        self,
        transform: Optional[Callable]
    ) -> None:
        """
        Get train and val dataloaders

        Parameters:
        ----------
        transform : Optional[Callable]
            Transform used on training dataset
        """

        # Get train/val data
        _data = torch.load(self.data_path)

        # Create and transform training dataset
        self.train_data = ImageLabelDataset(
            _data["train_images"],
            _data["train_labels"],
            transform = transform
        )
        self.val_data = ImageLabelDataset(
            _data["val_images"],
            _data["val_labels"]
        )
    
    def train_loader(self) -> None:

        # Create loader
        train_loader = create_loader(
            self.train_data,
            batch_size = self.batch_size,
            shuffle = True
        )
        return train_loader

    def val_loader(self) -> None:
        
        val_loader = create_loader(
            self.val_data,
            batch_size = self.batch_size,
            shuffle = False
        )
        return val_loader

    def create_dataloaders(self) -> None:
        """Create dict of train and val dataloaders"""

        self.dataloaders = {
            "train" : self.train_loader(),
            "val" : self.val_loader()
        }