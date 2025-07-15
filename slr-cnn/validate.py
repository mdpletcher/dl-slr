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
            self.preds = self.c.model(self.inputs)
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
        scheduler = ReduceLROnPlateau(
            self.c.optimizer,
            mode = "min",
            factor = 0.5,
            patience = config.PATIENCE,
            verbose = True,
            eps = 1e-4
        )
        scheduler.step(self.epoch_loss)

    def iterate_batches(self) -> None:
        """Iterate over batch in val dataloader and predict"""
        for self.batch, (inputs, labels) in enumerate(
            self.d.dataloaders["val"]
        ):
            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)
            self.labels = self.labels.view(-1, 1)

            # Make predictions
            self.predict()
            #print(self.loss)
            #print(self.preds)
            self.batch_metrics()
            #print(self.totals)
            self.append_preds()

    def run(self) -> torch.Tensor:
        self.iterate_batches()
        self.epoch_metrics()
        if not config.TUNE:
            self.reduce_lr()
        val_best_loss = self.save_model()
        #if config.TUNE:
        #    tune.report(loss = self.epoch_loss)
        self.print_epoch_metrics("val")
        return val_best_loss