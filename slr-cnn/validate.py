"""Validation metrics for models"""

import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List

from perf_metrics import Metrics
from config import TrainConfig
from model_config import ModelConfig
from data_setup import DataSetup
config = TrainConfig()



class Validation(Metrics):
    """
    Validation metrics for batched data

    Parameters:
    ----------
    d : data_setup.DataSetup
        DataSetup class instance
    epoch : int
        Epoch index in training loop
    epochs : int
        Total number of epoch in training loop
    val_best_loss : torch.Tensor
        Lowest loss across epochs
    c : model_config.ModelConfig
        ModelConfig class instance
    epoch_preds : List
        epoch predictions
    epoch_labels : List
        epoch labels
    """
    def __init__(
        self,
        d: DataSetup,
        epoch: int,
        epochs: int,
        val_best_loss: torch.Tensor,
        c: ModelConfig,
        epoch_preds: List = [],
        epoch_labels: List = []
    ):
        super().__init__(d, epoch, epochs)
        self.val_best_loss = val_best_loss
        self.c = c
        self.epoch_preds = epoch_preds
        self.epoch_labels = epoch_labels

    def predict(self) -> None:
        """Predict SLR and compute loss"""
        with torch.no_grad():
            self.preds = self.c.model(self.inputs, self.scalars)
            self.loss = self.c.criterion(self.preds, self.labels)

    def append_preds(self) -> None:
        """Append all predictions for each batch to an epoch"""
        self.epoch_preds.append(self.preds.cpu().tolist())
        self.epoch_labels.append(self.labels.cpu().tolist())

    def save_model(self) -> torch.Tensor:
        """Save best model weights after improvement in val loss"""

        if self.epoch_loss < self.val_best_loss:
            self.val_best_loss = self.epoch_loss
        elif self.epoch_loss < self.val_best_loss and config.SAVE_MODEL:
            print(
                "Epoch loss: %02d < best loss: %02d, saving model" % (self.epoch_loss, self.val_best_loss)
            )
            
            MODEL_SAVENAME = (
                "%sepoch%s_batch_size%s_%s_model.pt" % (
                    config.MODEL_SAVE_DIR, config.MAX_EPOCHS, config.BATCH_SIZE, config.MODEL_NAME
                )
            )
            torch.save(self.c.model, MODEL_SAVENAME)
        return self.val_best_loss


    def reduce_lr(self) -> None:
        """Reduce learning rate if epoch loss plateaus"""
        old_lr = self.c.optimizer.param_groups[0]["lr"]
        self.c.scheduler.step(self.epoch_loss)
        new_lr = self.c.optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"[Epoch {self.epoch}] LR reduced from {old_lr:.6e} to {new_lr:.6e}")

    def iterate_batches(self, lstm = False) -> None:
        """Iterate over batch in val dataloader and predict"""
        for self.batch, (inputs, scalars, labels) in enumerate(
            self.d.dataloaders["val"]
        ):
            self.inputs = inputs.to(config.DEVICE)
            self.scalars = scalars.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)
            self.labels = self.labels.view(-1, 1)

            # If using LSTM, permute inputs
            if lstm:
                self.inputs = self.inputs.permute(0, 3, 1, 2)

            # Make predictions
            self.predict()
            self.batch_metrics()
            self.append_preds()

    def run(self, lstm = False) -> torch.Tensor:
        self.iterate_batches(lstm = lstm)
        self.epoch_metrics()
        val_best_loss = self.save_model()
        #if config.TUNE:
        #    tune.report(loss = self.epoch_loss)
        self.print_epoch_metrics("val")
        return val_best_loss