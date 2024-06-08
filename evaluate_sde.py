import os
import json
import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge

from scipy.stats import norm

import torch

from models.sde import Generator, DiscriminatorSimple
from dataloaders import get_sde_dataloader

from statsmodels.tsa.stattools import acf, ccf


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_samples(data, columns, n_samples=10, dirname="plots/sde"):
    for i, var in enumerate(columns):
        plt.figure()
        lines = []
        for icolor, (k, v) in enumerate(data.items()):
            j = columns.index(var)
            lines.append(plt.plot(v[:n_samples, :, j].T, color=COLORS[icolor], linewidth=0.5, label=k))
        plt.ylabel(var)
        plt.xlabel("Hour of Day")
        plt.legend([l[0] for l in lines], list(data.keys()))
        plt.savefig(os.path.join(dirname, f"samples_{var}.png"), dpi=300)
        plt.close()


def plot_histograms(data, columns, is_diff=False, dirname="plots/sde"):
    for i, var in enumerate(columns):
        plt.figure()
        sample_min = np.inf
        sample_max = -np.inf
        for k, v in data.items():
            j = columns.index(var)
            sample_min = min(sample_min, np.min(v[..., j]))
            sample_max = max(sample_max, np.max(v[..., j]))
        bins = np.linspace(sample_min, sample_max, 50)
        values = []
        labels = []
        for k, v in data.items():
            values.append(v[..., j].flatten())
            labels.append(k)
        plt.hist(values, bins=bins, label=labels, density=True)
        # for k, v in data.items():
        #     plt.hist(v[..., j].flatten(), bins=bins, alpha=0.5, label=k, density=True)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(dirname, f"histogram_{'diff_' if is_diff else ''}{var}.png"), dpi=300)
        plt.close()


def plot_acf(data, columns, dirname="plots/sde"):
    for i, var in enumerate(columns):
        plt.figure()
        for k, v in data.items():
            samples_acf = np.array([acf(v[j, :, i], nlags=12, fft=False) for j in range(v.shape[0])])
            plt.plot(np.arange(13), samples_acf.mean(axis=0), label=k)
            plt.fill_between(np.arange(13), np.percentile(samples_acf, 2.5, axis=0), np.percentile(samples_acf, 97.5, axis=0), alpha=0.5)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.savefig(os.path.join(dirname, f"acf_{var}.png"), dpi=300)


def plot_xcf(data, columns, dirname="plots/sde"):
    for idx_var1, var in enumerate(columns):
        for idx_var2, var2 in enumerate(columns):
            if idx_var1 <= idx_var2:  # No auto-correlation, no repeated pairs
                continue
            plt.figure()
            for k, v in data.items():
                backwards = np.array([ccf(v[i, :, idx_var1][::-1], v[i, :, idx_var2][::-1], nlags=13, fft=False)[::-1] for i in range(v.shape[0])])
                forwards  = np.array([ccf(v[i, :, idx_var1],       v[i, :, idx_var2],       nlags=13, fft=False)       for i in range(v.shape[0])])
                xcf = np.concatenate([backwards[:, :-1], forwards], axis=1)
                plt.plot(np.arange(-12, 13), xcf.mean(axis=0), label=k)
                plt.fill_between(np.arange(-12, 13), np.percentile(xcf, 2.5, axis=0), np.percentile(xcf, 97.5, axis=0), alpha=0.5)
            plt.xlabel("Lag")
            plt.ylabel("Cross-Correlation")
            plt.title(f"{var} vs {var2}")
            plt.legend()
            plt.savefig(os.path.join(dirname, "ccf_{var}_{var2}.png"), dpi=300)


def stack_wedges(ax, base_x: float, base_y: float, u: float, v: float, std: float, n: int = 1, scale: float = 0.25):
    """
    Mimic a gradient by stacking n wedges with low opacity on top of each other. The width of
    the wedges follows a normal distribution.
    """
    assert n > 0
    if n == 1:  # 95% confidence interval
        z_scores = [norm.ppf(0.975)]
    elif n == 2:  # 50% and 95% confidence intervals
        z_scores = [norm.ppf(0.75), norm.ppf(0.975)]
    else:  # n confidence intervals between 50% and 95%
        z_scores = norm.ppf(np.linspace(0.5, 0.975, n + 1))[1:]  # Skip the first one, which will have width 0

    std = min(np.abs(std), 1e-6)

    for z in z_scores:
        theta1 = np.rad2deg(np.arctan2(v - z * std, u))
        theta2 = np.rad2deg(np.arctan2(v + z * std, u))
        # theta1, theta2 = sorted([theta1, theta2])
        ax.add_patch(Wedge((base_x, base_y), scale, theta1, theta2, color="red", alpha=1/(n+1)))


def plot_gradients(model: Generator, init_noise, varnames: list[str], transformer, dirname="plots/sde"):
    # Sample the model
    samples, drift, diffusion = model(init_noise, gradients=True)
    samples = samples.detach().cpu().numpy()
    drift = drift.detach().cpu().numpy()
    diffusion = diffusion.detach().cpu().numpy()

    # We know our only scaling is StandardScaler for TOTALLOAD. We'll rescale things manually so we
    # don't mess up our gradients.
    mean = transformer.transformers["TOTALLOAD"].mean.item()
    std  = transformer.transformers["TOTALLOAD"].std.item()
    load_idx = varnames.index("TOTALLOAD")
    samples[:, :, load_idx] = samples[:, :, load_idx] * std + mean
    drift[:, :, load_idx] *= std
    diffusion[:, :, load_idx] *= std

    # Plot a few samples along with arrows indicating the drift direction and a colored background
    # indicating the diffusion.
    fig, ax = plt.subplots(nrows=len(varnames), ncols=1, figsize=(15, 10))
    ts = np.arange(24)
    scale = 0.5
    for isample in range(5):
        for ivar, varname in enumerate(varnames):
            ax[ivar].plot(ts, samples[isample, :, ivar], color="blue")
            for t in range(len(ts)):
                # Layer several wedges on top of each other to mimic a gradient matching a normal distribution
                base_x = ts[t]
                base_y = samples[isample, t, ivar]
                u = 1.0  # dt/dt = 1.0 everywhere
                v = drift[isample, t, ivar]
                stack_wedges(ax[ivar], base_x, base_y, u, v, diffusion[isample, t, ivar], 10, scale=scale)

    if isinstance(model, Generator):
        fig.savefig(os.path.join(dirname, "gradients.png"), dpi=300)
    else:
        fig.savefig(os.path.join(dirname, "gradients_swa.png"), dpi=300)
    plt.close(fig)


def plot_model_results(
    G: Generator,
    transformer,
    varnames: list[str],
    G_swa: Generator | None = None,
    n_samples: int = 1826,
    dir_suffix: str = ""
):
    init_noise = G.sample_latent(n_samples)
    samples = G(init_noise)
    sde_samples = pd.DataFrame(np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy()), columns=varnames)
    sde_samples.to_csv(f"ercot_samples_sde.csv", index=False)

    data_locations = {
        "Historical": ("dataloaders/data/ercot.csv", dict(index_col=0)),
        "DGAN": ("ercot_samples_dgan.csv", dict()),
    }

    data = {}
    for k, (fpath, kwargs) in data_locations.items():
        if os.path.exists(fpath):
            data[k] = pd.read_csv(fpath, **kwargs)
    data["SDE"] = sde_samples
    if G_swa is not None:
        samples_swa = G_swa(init_noise)
        sde_samples_swa = pd.DataFrame(np.vstack(transformer.inverse_transform(samples_swa).detach().cpu().numpy()), columns=varnames)
        sde_samples_swa.to_csv(f"ercot_samples_sde_swa.csv", index=False)
        data["SDE_SWA"] = sde_samples_swa

    if "Historical" in data:
        data["Historical"].pop("PRICE")
    data = {k: v.values.reshape(-1, 24, len(varnames)) for k, v in data.items()}

    if "Historical" in data:
        np.random.shuffle(data["Historical"])

    suffix = dir_suffix if dir_suffix == "" else f"_{dir_suffix}"
    dirname = f"plots/sde{suffix}"
    os.makedirs(dirname, exist_ok=True)

    # Plot the samples, with each variable in a separate plot
    plot_samples(data, varnames, dirname=dirname)

    # Plot a histogram of the samples. The bars of the generated and historical data should be side
    # by side, not stacked or overlapping.
    plot_histograms(data, varnames, dirname=dirname)

    # Plot a histogram of the first differences of the samples. Again, we use side by side bars.
    first_diffs = {k: np.diff(v, axis=1) for k, v in data.items()}
    plot_histograms(first_diffs, varnames, is_diff=True, dirname=dirname)

    # Calculate the autocorrelation of the samples and the historical data, calculated for each
    # sample. The mean and 95% confidence intervals are plotted. We use the first N//2 lags.
    plot_acf(data, varnames, dirname=dirname)

    # Calculate the lagged cross-correlation between each variable for the generated and historical
    # data. Plot lags -N//2 to N//2.
    plot_xcf(data, varnames, dirname=dirname)

    # Plot the gradients of the model. This is a bit more involved, as we need to sample the model
    # and calculate the gradients of the drift and diffusion functions.
    grad_init_noise = G.sample_latent(128)
    plot_gradients(G, grad_init_noise, varnames, transformer, dirname=dirname)
    plot_gradients(G_swa, grad_init_noise, varnames, transformer, dirname=dirname)
