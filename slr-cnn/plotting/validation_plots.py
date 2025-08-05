import numpy as np
import pandas as pd

from copy import copy
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import r2_score
from typing import Dict, Optional, List

class VerificationPlots():

    """
    Verification plots for ML models
    """

    def __init__(
        self,
        preds,
        obs,
        fname,
        fsave = False
    ):
        self.preds = preds
        self.obs = obs
        self.fname = fname
        self.fsave = fsave
    
    def compute_metrics(self) -> Dict[str, float]:

        MAE = np.nanmean(abs(self.preds - self.obs))
        R2 = r2_score(self.obs, self.preds)
        RMSE = np.sqrt(np.nanmean((self.obs - self.preds) ** 2))
        MBE = np.nanmean(self.preds - self.obs)

        return {
            "MAE" : MAE,
            "R2" : R2,
            "RMSE" : RMSE,
            "MBE" : MBE
        }

    def histogram(self, bin_width, max_val) -> None:
        self.bins = np.arange(0, max_val, bin_width)
        self.heatmap, self.xedges, self.yedges = np.histogram2d(self.obs, self.preds, bins = self.bins)
        
    def get_colormap(self, cmap_name, levels) -> None:
        cmap = copy(plt.get_cmap(cmap_name))
        cmap.set_under("w", 1)
        self.cmap = cmap
        levels = levels.copy()
        levels[0] = 1e-5
        self.norm = BoundaryNorm(levels, ncolors = self.cmap.N, clip = True)

    def plot_bin_medians(self, ax) -> None:

        fcst_label_bool = False
        obs_label_bool = False

        for i, rowbin in enumerate(self.bins[1:]):
            fc_med = np.nanmedian(
                np.where(
                    (self.obs > rowbin - 2.5) & (self.obs <= rowbin), 
                    self.preds, 
                    np.nan
                )
            )
            ob_med = np.nanmedian(
                np.where(
                    (self.preds > rowbin - 2.5) & (self.preds <= rowbin), 
                    self.obs, 
                    np.nan
                )
            )
            no_obs = np.where(
                (self.obs > rowbin - 2.5) & (self.obs <= rowbin), 
                self.preds, 
                np.nan
            )
            no_obs = no_obs[~np.isnan(no_obs)]

            if len(no_obs) >= 10:
                if ~np.isnan(fc_med):
                    ax.scatter(
                        np.searchsorted(self.bins, fc_med) - 0.5, 
                        i + 0.5,
                        zorder = 100, 
                        c = "white", 
                        s = 22.5, 
                        edgecolor = "k", 
                        marker = "o",
                        label = "Forecast Median" if not fcst_label_bool else ""
                    )
                    fcst_label_bool = True
                if ~np.isnan(ob_med):
                    ax.scatter(
                        i + 0.5, 
                        np.searchsorted(self.bins, ob_med) + 0.5,
                        zorder = 100, 
                        c = "gray", 
                        s = 22.5, 
                        edgecolor = "k", 
                        marker = "o",
                        label = "Observed Median" if not obs_label_bool else ""
                    )
                    obs_label_bool = True
    
    def plot_format(self, ax, max_val, plot_type = "scatter") -> None:
        if plot_type == "scatter":
            ax.set_facecolor("lightgray")
            ax.grid(color = "w")
            ax.margins(x = 0.5)
            ax.tick_params(axis = "both", which = "major", labelsize = 16, pad = 5)
            ax.set_xticks(np.arange(0, max_val + 1, 5))
            ax.set_yticks(np.arange(0, max_val + 1, 5))

            # Legend
            leg = ax.legend(loc = "lower right", prop = {"size" : 16})
            leg.get_title().set_fontsize(16)
            leg.get_frame().set_edgecolor("k")
            leg.get_frame().set_boxstyle("Square")
            leg.get_frame().set_linewidth(0.75)

        elif plot_type == "hist_2d":
            ax.set_xticks(np.arange(0, max_val + 1, 5))
            ax.set_yticks(np.arange(0, max_val + 1, 5))
            ax.axline((0, 0), (1, 1), linewidth = 0.5, color = "k", zorder = 1)
            ax.grid(True, which = "major", linestyle = "-", color = "k", alpha = 0.5)
            ax.grid(True, which = "minor", color = "k", linestyle = "-", alpha = 0.5)
            ax.tick_params(axis = "both", which = "major", labelsize = 18, pad = 5)
            ax.set_aspect("equal", adjustable = "box")
            ax.set_ylabel("Observed", fontsize = 24)
            ax.set_xlabel("Predicted", fontsize = 24)
            ax.set_yticklabels(np.arange(0, max_val + 1, 5))
            ax.set_xticklabels(np.arange(0, max_val + 1, 5))
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)

    def plot_metrics_text(self, ax) -> None:
        self.metrics = self.compute_metrics()
        text = ax.text(
            0.665, 0.0325,
            (
                f"$\\mathregular{{R^2}}$: {self.metrics['R2']:.2f}\n"
                f"MAE: {self.metrics['MAE']:.2f}\n"
                f"RMSE: {self.metrics['RMSE']:.2f}\n"
                f"MBE: {self.metrics['MBE']:.2f}\n"
                f"n = {len(self.preds)}"
            ),
            fontsize = 24,
            transform = ax.transAxes,
            bbox = dict(facecolor = 'white', alpha = 0.6, pad = 4.0)
        )

    def add_hist_colorbar(self, fig, hist_plot):
        cbar_ax = fig.add_axes([0.085, -0.0425, 0.95, 0.05])
        cbar = fig.colorbar(
            hist_plot, 
            cax = cbar_ax, 
            fraction = 0.046, 
            pad = 0.04,
            orientation = 'horizontal', 
            extend = 'max'
        )
        cbar.set_label(label = '# of events', size = 24)
        cbar.ax.tick_params(labelsize = 18)

    def hist_2d(
        self, 
        bin_width = 1,
        max_val = 40,
        levels: Optional[List[float]] = None,
        cmap_name = "plasma",
        plot_medians = False,
        cbar = True,
        title = None
    ):
        fig, ax = plt.subplots(figsize = (7.5, 7.5))
        self.histogram()
