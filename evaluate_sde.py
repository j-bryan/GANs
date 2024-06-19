import os
import json
import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge

from scipy.stats import norm
from scipy.stats import wasserstein_distance

import torch

from models.sde import Generator, DiscriminatorSimple
from dataloaders import get_sde_dataloader

from statsmodels.tsa.stattools import acf, ccf


# COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_samples(data, columns, n_samples=6, dirname="plots/sde"):
    for i, var in enumerate(columns):
        nrows = np.floor(np.sqrt(n_samples)).astype(int)
        ncols = np.ceil(n_samples / nrows).astype(int)
        n_samples = nrows * ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

        # Plot one sample from each item of data in each subplot. Find the minimum and maximum value
        # of the "Historical" data sample and find a sample from the generated data that is closest
        # to that range. This will hopefully make the samples more comparable.

        for irow in range(nrows):
            for icol in range(ncols):
                ax[irow, icol].set_xlabel("Hour of Day")
                ax[irow, icol].set_ylabel(var)

                historical_sample = data["Historical"][np.random.randint(data["Historical"].shape[0]), :, i]
                ax[irow, icol].plot(historical_sample, label="Historical", color=COLORS[0])
                hist_min = np.min(historical_sample)
                hist_max = np.max(historical_sample)

                print("Num values in data", len(data))
                print("Num colors in list", len(COLORS))
                for icolor, (k, v) in enumerate(data.items()):
                    if k == "Historical":
                        continue
                    # Find the sample that is closest to the range of the historical data
                    sample_idx = np.argmin([np.abs(np.min(v[j, :, i]) - hist_min) + np.abs(np.max(v[j, :, i]) - hist_max) for j in range(v.shape[0])])
                    ax[irow, icol].plot(v[sample_idx, :, i], color=COLORS[icolor], linewidth=0.5, label=k)

        # for icolor, (k, v) in enumerate(data.items()):
        #     j = columns.index(var)
        #     plt.plot(v[:n_samples, :, j].T, color=COLORS[icolor], linewidth=0.5, label=k)

        plt.ylabel(var)
        plt.xlabel("Hour of Day")
        plt.legend()
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
        bins = np.linspace(sample_min, sample_max, 25)
        values = []
        labels = []
        for k, v in data.items():
            values.append(v[..., j].flatten())
            labels.append(k)
        plt.hist(values, bins=bins, label=labels, density=True)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(dirname, f"histogram_{'diff_' if is_diff else ''}{var}.png"), dpi=300)
        plt.close()

        import plotly.figure_factory as ff

        # Group data together
        hist_data = []
        group_labels = []
        min_val = None
        max_val = None
        key_order = ["Historical", "ARMA", "CNN", "DGAN", "SDE"]
        for k in key_order:
            if k not in data:
                continue
            v = data[k]
            j = columns.index(var)
            vals = v[..., j].flatten()
            hist_data.append(vals)
            group_labels.append(k)
            min_val = min(vals) if min_val is None else min(min_val, min(vals))
            max_val = max(vals) if max_val is None else max(max_val, max(vals))
        bin_size = (max_val - min_val) / 25

        # Create distplot with custom bin_size
        # fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size, show_hist=False)
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
        fig.write_image(os.path.join(dirname, f"histogram_{'diff_' if is_diff else ''}{var}_plotly.png"))


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
            plt.savefig(os.path.join(dirname, f"ccf_{var}_{var2}.png"), dpi=300)


def stack_wedges(ax, base_x: float, base_y: float, u: float, v: float, diffusion: float | np.ndarray, n: int = 1, scale: float = 0.25):
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

    # diffusion may be a float or a (N,) array. If it's a float, we're dealing with an SDE with "diagonal"
    # noise. If it's an array, we're dealing with "general" noise. Note that we've defined a Stratonovich
    # SDE, and Stratonovich integrals are not always zero-mean! We need to convert the stratono
    if isinstance(diffusion, np.ndarray):
        pass
    elif isinstance(diffusion, float):
        pass
    else:
        raise ValueError("diffusion must be a float or a numpy array")
    # std = min(np.abs(std), 1e-6)

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

    # The drift and diffusion values correspond to the Stratonovich SDE
    #     dY_t = mu(Y_t, t) dt + sigma(Y_t, t) \circ dW_t
    # To understand the gradients, we need to plot the drift and diffusion values. While the drift
    # is easily interpreted, the diffusion is a bit more tricky due to the Stratonovich integral. We
    # need to convert the diffusion to an Ito integral to make sense of it, since the Ito integral
    # has nice properties like zero mean.
    # TODO!

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
    G: Generator | None = None,
    transformer=None,
    varnames: list[str] = ["TOTALLOAD", "WIND", "SOLAR"],
    dirname: str = "",
    G_swa: Generator | None = None,
    n_samples: int = 1826
):
    device = "cpu"

    if G_swa is not None:
        G_swa = G_swa.to(device)

    if G is not None:
        G = G.to(device)
        init_noise = G.sample_latent(n_samples)

        samples = G(init_noise, time_steps=24)
        samples = np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy())
        if samples.shape[-1] > len(varnames):  # drop
            samples = samples[..., -len(varnames):]
        sde_samples = pd.DataFrame(np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy()), columns=varnames)
        sde_samples.to_csv(f"ercot_samples_sde_varlength.csv", index=False)

        samples168 = G(init_noise, time_steps=168)
        samples168 = np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy())
        if samples168.shape[-1] > len(varnames):  # drop
            samples168 = samples168[..., -len(varnames):]
        sde_samples168 = pd.DataFrame(np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy()), columns=varnames)
        sde_samples168.to_csv(f"ercot_samples_sde_varlength168.csv", index=False)

    data_locations = {
        "Historical": ("dataloaders/data/ercot.csv", dict(index_col=0)),
        "ARMA": ("ercot_samples_arma.csv", dict()),
        "CNN": ("ercot_samples_cnn.csv", dict()),
        "DGAN": ("ercot_samples_dgan.csv", dict()),
        "SDE": ("ercot_samples_sde.csv", dict())
    }

    data = {}
    for k, (fpath, kwargs) in data_locations.items():
        if os.path.exists(fpath):
            data[k] = pd.read_csv(fpath, **kwargs)

    if G is not None:
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

    os.makedirs(dirname, exist_ok=True)
    if G_swa is not None:
        swa_dirname = os.path.join(dirname, "swa/")
        os.makedirs(swa_dirname, exist_ok=True)

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
    # grad_init_noise = G.sample_latent(128)
    # plot_gradients(G, grad_init_noise, varnames, transformer, dirname=dirname)
    # plot_gradients(G_swa, grad_init_noise, varnames, transformer, dirname=dirname)
    # plot_gradients(G_swa, grad_init_noise, varnames, transformer, dirname=swa_dirname)


def calculate_metrics(G, historical, transformer, varnames, G_swa=None):
    metrics = {}

    # Sample the model
    if isinstance(historical, pd.DataFrame):
        historical = historical[varnames].values
    elif isinstance(historical, torch.Tensor):
        historical = historical.detach().cpu().numpy()
    elif isinstance(historical, torch.utils.data.DataLoader):
        historical = historical.dataset.detach().cpu().numpy()

    n_samples = len(historical)
    latent = G.sample_latent(n_samples)
    samples_G = G(latent)
    samples_G = transformer.inverse_transform(samples_G).detach().cpu().numpy()
    if samples_G.ndim == 3:
        samples_G = samples_G.reshape(-1, 24, len(varnames))

    if G_swa is not None:
        samples_swa = G_swa(latent)
        samples_swa = transformer.inverse_transform(samples_swa).detach().cpu().numpy()
        if samples_swa.ndim == 3:
            samples_swa = samples_swa.reshape(-1, 24, len(varnames))

    samples = {
        "raw": samples_G,
        "swa": samples_swa
    }

    for i, var in enumerate(varnames):
        for model_name, generated in samples.items():
            # Wasserstein distance of the raw data
            var_scale = np.ptp(historical[..., i].ravel())
            metrics[f"wd_{model_name}_{var}"] = wasserstein_distance(historical[..., i].ravel(), generated[..., i].ravel()) / var_scale

            # Wasserstein distance of the first differences
            max_diff = np.ptp(np.diff(historical, axis=1).ravel())
            metrics[f"wd_diff1_{model_name}_{var}"] = wasserstein_distance(np.diff(historical, axis=1).ravel(), np.diff(generated, axis=1).ravel()) / (2 * max_diff)

            # Wasserstein distance of the autocorrelation function by hour, summed over hours
            acf_historical = np.array([acf(historical[j, :, i], nlags=12, fft=False) for j in range(historical.shape[0])])
            acf_generated = np.array([acf(generated[j, :, i], nlags=12, fft=False) for j in range(generated.shape[0])])
            acf_wd = 0
            for j in range(acf_historical.shape[1]):
                acf_wd += wasserstein_distance(acf_historical[:, j], acf_generated[:, j])
            metrics[f"wd_acf_{model_name}_{var}"] = acf_wd / (2 * acf_generated.shape[1])  # maximum peak-to-peak distance is 2 (acf ranges from -1 to 1)

            # Wasserstein distance of the cross-correlation function by hour, summed over hours
            for j, var2 in enumerate(varnames):
                if i <= j:
                    continue
                wd_xcf = 0

                backwards_hist = np.array([ccf(historical[k, :, i][::-1], historical[k, :, j][::-1], nlags=13, fft=False)[::-1] for k in range(historical.shape[0])])
                forwards_hist  = np.array([ccf(historical[k, :, i],       historical[k, :, j],       nlags=13, fft=False)       for k in range(historical.shape[0])])
                xcf_historical = np.concatenate([backwards_hist[:, :-1], forwards_hist], axis=1)

                backwards_gen  = np.array([ccf(generated[k, :, i][::-1], generated[k, :, j][::-1], nlags=13, fft=False)[::-1] for k in range(generated.shape[0])])
                forwards_gen   = np.array([ccf(generated[k, :, i],       generated[k, :, j],       nlags=13, fft=False)       for k in range(generated.shape[0])])
                xcf_gen        = np.concatenate([backwards_gen[:, :-1], forwards_gen], axis=1)

                for k in range(xcf_historical.shape[1]):
                    wd_xcf += wasserstein_distance(xcf_historical[:, k], xcf_gen[:, k])

                metrics[f"wd_xcf_{model_name}_{var}_{var2}"] = wd_xcf / (2 * xcf_gen.shape[1])  # maximum peak-to-peak distance is 2 (xcf ranges from -1 to 1)

    metrics["total"] = sum([v for k, v in metrics.items() if "wd_" in k])

    return metrics


if __name__ == "__main__":
    fire.Fire({
        "plot": plot_model_results,
        "metrics": calculate_metrics
    })
