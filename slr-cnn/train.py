"""Class for training DL models"""
import os

import numpy as np
import torch

from perf_metrics import Metrics
from data_setup import DataSetup
from model_config import ModelConfig
from config import TrainConfig
config = TrainConfig()

class Train(Metrics):
    """
    Train using a batched dataset

    Parameters:
    ----------
    d : data_setup.DataSetup 
        DataSetup class instance
    epoch : int
        Training loop epoch index
    epochs : int
        Total # of epochs in training loop
    c : (model_config.ModelConfig) : ModelConfig class instance

    Inspired by Vanessa Przybylo's COCPIT repo
    https://github.com/vprzybylo/cocpit/blob/master/cocpit/train.py
    """

    def __init__(
        self,
        d: DataSetup,
        epoch: int,
        epochs: int,
        c: ModelConfig
    ):
        super().__init__(d, epoch, epochs)
        self.c = c
    
    def train_step(self) -> None:
        """
        Makes predictions, evaluates loss, performs backpropagation,
        and updates model parameters
        """

        outputs = self.c.model(self.inputs, self.scalars)
        self.loss = self.c.criterion(outputs, self.labels)
        self.loss.backward()
        self.c.optimizer.step()

    def iterate_batches(self, lstm = False) -> None:
        """
        Iterate over DataLoader batch and train
        """
        for self.batch, (inputs, scalars, labels) in enumerate(
            self.d.dataloaders["train"]
        ):
            self.inputs = inputs.to(config.DEVICE)
            self.scalars = scalars.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)
            self.labels = self.labels.view(-1, 1)

            self.c.optimizer.zero_grad()
            # If using LSTM, permute inputs
            if lstm:
                self.inputs = self.inputs.permute(0, 3, 1, 2)
            self.train_step()
            self.batch_metrics()

    def run(self, lstm = False) -> None:
        self.iterate_batches(lstm = lstm)
        self.epoch_metrics()