"""Model configuration for the SLR CNN"""

import torch
from torch import optim, nn

from config import TrainConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

config = TrainConfig()

class ModelConfig:

    """
    Configuration that sets the model optimizer, loss function,
    dropout, and whether to utilize CPU or GPU.

    Parameters:
    ----------
    model : torch.nn.Module
        PyTorch model object
    optimizer : 
        PyTorch built-in optimizer for updating model weights
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer: torch.optim.Optimizer = None

    def n_parameters(self) -> int:
        """Number of trainable model parameters"""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def set_optimizer(
        self, 
        lr = 0.001, 
        weight_decay = 0.0, 
        optimizer_type = 'adam'
    ) -> None:
        """Set model optimizer to account for nonlinear relationships"""

        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr = lr,
                weight_decay = weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr = lr,
                weight_decay = weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = lr,
                weight_decay = weight_decay,
                momentum = 0.9
            )
        elif optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr = lr,
                weight_decay = weight_decay
            )
        
        # Reduce learning rate if validation loss plateaus 
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode = "min",
            factor = 0.5,
            patience = 3,
            eps = 1e-4
        )

    def set_criterion(self, criterion) -> None:
        """Sets loss function criteria"""
        if criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "MAE":
            self.criterion = nn.L1Loss()

    def to_device(self) -> None:
        """Train with GPUs if available"""
        if torch.cuda.device_count() > 1:
            print("Using %s GPUs" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(config.DEVICE)
            