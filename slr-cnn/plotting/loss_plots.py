"""
Script from https://github.com/vprzybylo/cocpit/blob/cfdfe484ece483d33b5a45a2edc8b90a1466d0bd/cocpit/plotting_scripts/acc_loss.py
Edited by Michael Pletcher
"""

import matplotlib.pyplot as plt
import numpy as np

class LossPlots:
    """
    Plot training and validation losses per epoch
    """

    def __init__(
        self,
        model_type,
        num_epochs,
        fname,
        colors,
        fsave = False
    ):
        self.model_type = model_type
        self.num_models = len(model_type)
        self.num_epochs = num_epochs
        self.fname = fname
        self.colors = colors
        self.fsave = fsave

    def plot_loss(
        self,
        ax,
        loss,
        marker,
        size,
        zorder = 1
    ):

        # Plot data for each model
        for i in range(self.num_models):
            print()
            ax.scatter(
                np.arange(1, (self.num_epochs + 1)),
                loss[i],
                c = self.colors[i],
                marker = marker,
                s = size,
                label = self.model_type[i],
                zorder = 10
            )
            ax.plot(
                np.arange(1, (self.num_epochs + 1)),
                loss[i],
                c = self.colors[i],
                zorder = 11,
                linewidth = 2
            )

    def plot_format(self, ax, label):

        # Plot
        ax.set_facecolor("lightgray")
        ax.grid(color = "w")
        ax.margins(x = 0.5)
        ax.set_xlabel("Number of epochs", fontsize = 20)
        ax.set_ylabel("Loss (MSE)", fontsize = 20)
        ax.tick_params(axis = "both", which = "major", labelsize = 16, pad = 5)
        ax.set_xticks(np.arange(1, 15 + 1, 1))
        ax.set_yticks(np.arange(0, 150 + 1, 10))

        # Legend
        leg = ax.legend(loc = "upper right", prop = {"size" : 16})
        leg.get_title().set_fontsize(16)
        leg.get_frame().set_edgecolor("k")
        leg.get_frame().set_boxstyle("Square")
        leg.get_frame().set_linewidth(0.75)

    def training_loss(self, ax, train_loss):
        self.plot_loss(ax, loss = train_loss, marker = "o", size = 45)
        self.plot_format(ax, label = "Training")
        ax.set_title("Training", fontsize = 24)
        ax.set_ylim(0, 25)
        ax.set_xlim(1, self.num_epochs)

    def validation_loss(self, ax, val_loss):
        self.plot_loss(ax, loss = val_loss, marker = "o", size = 45)
        self.plot_format(ax, label = "Validation")
        ax.set_title("Validation", fontsize = 24)
        ax.set_ylim(0, 25)
        ax.set_xlim(1, self.num_epochs)

    def save(self):
        plt.savefig(self.fname, dpi = 750, bbox_inches = "tight")
    
    def make_plot(self, train_loss, val_loss):
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize = (8, 10))
        self.training_loss(ax1, train_loss)
        self.validation_loss(ax2, val_loss)
        plt.tight_layout()
        if self.fsave:
            self.save()