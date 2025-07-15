"""
Calculates loss metrics for epochs and batches for the training
and validation datasets
"""

import torch

from config import TrainConfig
config = TrainConfig()

class Metrics:
    """
    Calculate batch/epoch metrics for train/val datasets

    Parameters:
    ----------
    d : data_setup.DataSetup
        DataSetup class instance
    epoch : int
        Epoch index in training loop
    epochs : int
        Total number of epoch in training loop

    """
    def __init__(self, d, epoch, epochs):

        """
        Initialize stuff
        """

        self.d: data_setup.DataSetup = d
        self.epoch: int = epoch
        self.epochs: int = epochs

        default_torch_type = torch.Tensor([0.0], device = config.DEVICE)
        self.loss: torch.Tensor = default_torch_type
        self.preds: torch.Tensor = default_torch_type
        self.inputs: torch.Tensor = default_torch_type
        self.labels: torch.Tensor = default_torch_type
        self.batch: int = 0

        self.totals: int = 0
        self.running_loss: float = 0.0
        self.running_corrects: torch.Tensor = default_torch_type
        self.batch_acc: float = 0.0
        self.epoch_loss: float = 0.0
        self.epoch_acc: torch.Tensor = default_torch_type

    def batch_metrics(self) -> None:
        """Batch loss"""
        #print(self.loss.item())
        #print(self.inputs.size(0))
        #print(self.loss.item())
        #print(self.inputs.size(0))
        #print(self.labels.size(0))
        self.running_loss += self.loss.item() * self.inputs.size(0)
        self.totals += self.labels.size(0)

    def epoch_metrics(self) -> None:
        """
        Calculate loss over all batches for an epoch
        """
        #print(self.running_loss)
        #print(self.totals)
        self.epoch_loss = self.running_loss / self.totals

    def print_batch_metrics(self, phase: str) -> None:
        """
        Print batch iteration and loss

        Parameters:
        ----------
        phase : str
            "train" or "val"
        """
        #print(phase)
        #print(self.batch + 1)
        #print(len(self.d.data_loaders[phase]))
        #print(self.loss.item())
        print(
            "%s, Batch %s/%s, Loss: %03f" % (
                phase, self.batch + 1, len(self.d.dataloaders[phase]), self.loss.item()
            )
        )

    def print_epoch_metrics(self, phase: str) -> None:
        """
        Print epoch loss for specified phase

        Parameters:
        ----------
        phase : str
            "train" or "val"
        """
        #print(phase)
        #print(self.epoch + 1)
        #print(self.epochs)
        #print(self.epoch_loss)
        print(
            "%s phase, Epoch %s/%s, Loss: %03f" % (
                phase, self.epoch + 1, self.epochs, self.epoch_loss
            )
        )