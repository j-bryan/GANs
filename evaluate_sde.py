import os
import json
import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge
matplotlib.use("TkAgg")

from scipy.stats import norm

import torch

from models.forcing import DawnDuskSampler
from models.sde_no_embed import SampledInitialCondition
# from models.sde_statespace import Generator, DiscriminatorSimple
from models.sde import Generator, DiscriminatorSimple
from dataloaders import get_sde_dataloader

from statsmodels.tsa.stattools import acf, ccf


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_samples(data, columns, n_samples=10):
    for i, var in enumerate(columns):
        plt.figure()
        lines = []
        for icolor, (k, v) in enumerate(data.items()):
            j = columns.index(var)
            lines.append(plt.plot(v[:n_samples, :, j].T, color=COLORS[icolor], linewidth=0.5, label=k))
        plt.ylabel(var)
        plt.xlabel("Hour of Day")
        plt.legend([l[0] for l in lines], list(data.keys()))
        plt.savefig(f"plots/sde/samples_{var}.png", dpi=300)
        plt.close()


def plot_histograms(data, columns, is_diff=False):
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
        plt.savefig(f"plots/sde/histogram_{'diff_' if is_diff else ''}{var}.png", dpi=300)
        plt.close()


def plot_acf(data, columns):
    for i, var in enumerate(columns):
        plt.figure()
        for k, v in data.items():
            samples_acf = np.array([acf(v[j, :, i], nlags=12, fft=False) for j in range(v.shape[0])])
            plt.plot(np.arange(13), samples_acf.mean(axis=0), label=k)
            plt.fill_between(np.arange(13), np.percentile(samples_acf, 2.5, axis=0), np.percentile(samples_acf, 97.5, axis=0), alpha=0.5)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.savefig(f"plots/sde/acf_{var}.png", dpi=300)


def plot_xcf(data, columns):
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
            plt.savefig(f"plots/sde/ccf_{var}_{var2}.png", dpi=300)


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


def plot_gradients(model: Generator, varnames: list[str], transformer):
    # Sample the model
    samples, drift, diffusion = model.sample(128, gradients=True)
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

    plt.show()


def make_plots():
    data = {
        "Historical": pd.read_csv("dataloaders/data/ercot.csv", index_col=0),
        "SDE": pd.read_csv("ercot_samples_sde.csv"),
        "DGAN": pd.read_csv("ercot_samples_dgan.csv")
    }
    data["Historical"].pop("PRICE")
    columns = list(data["Historical"].columns)
    data = {k: v.values.reshape(-1, 24, len(columns)) for k, v in data.items()}
    np.random.shuffle(data["Historical"])
    os.makedirs("plots/sde", exist_ok=True)

    # Plot the samples, with each variable in a separate plot
    plot_samples(data, columns)

    # Plot a histogram of the samples. The bars of the generated and historical data should be side
    # by side, not stacked or overlapping.
    plot_histograms(data, columns)

    # Plot a histogram of the first differences of the samples. Again, we use side by side bars.
    first_diffs = {k: np.diff(v, axis=1) for k, v in data.items()}
    plot_histograms(first_diffs, columns, is_diff=True)

    # Calculate the autocorrelation of the samples and the historical data, calculated for each
    # sample. The mean and 95% confidence intervals are plotted. We use the first N//2 lags.
    plot_acf(data, columns)

    # Calculate the lagged cross-correlation between each variable for the generated and historical
    # data. Plot lags -N//2 to N//2.
    plot_xcf(data, columns)


def main(
    params_file: str,
    n_samples: int = 1826,
):
    with open(params_file, 'r') as f:
            params = json.load(f)

    # Find the most appropriate device for training
    device = "cpu"

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]

    # seed for reproducibility
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])

    # Load the data from file and return as a DataLoader. Some additional transformations may have
    # been applied to the data and are returned as well.
    dataloader, transformer = get_sde_dataloader(iso=params['ISO'],
                                                 varname=params['variables'],
                                                 segment_size=params['time_series_length'],
                                                 batch_size=params['batch_size'],
                                                 device=device)

    # Some of the data is bad! We define that solar values must be 0 at night, which we determine to
    # be the hours 0, 1, 2, 3, 4, 21, 22, and 23. We'll set the values at these time steps to 0.
    solar = dataloader.dataset[..., params["variables"].index("SOLAR")]
    solar[:, [0, 1, 2, 3, 4, 21, 22, 23]] = 0.0
    dataset = dataloader.dataset
    dataset[..., params["variables"].index("SOLAR")] = solar
    # Write these values to file to load more easily for plotting
    historical_samples = pd.DataFrame(np.vstack(dataset.cpu().numpy()), columns=params['variables'])
    historical_samples.to_csv(f"ercot_samples_historical.csv", index=False)
    # Remake the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    params["batch_size"] = len(dataloader)
    dawn_dusk_sampler = DawnDuskSampler(solar)

    # The initial condition for the main state variables are sampled directly from the data.
    initial_condition = SampledInitialCondition(data=dataloader.dataset)
    state_size = len(params['variables'])

    G = Generator(initial_condition=initial_condition,
                  time_size=params['time_series_length'],
                  state_size=state_size,
                  noise_size=params['noise_size'],
                  f_num_units=params['gen_f_num_units'],
                  f_num_layers=params['gen_f_num_layers'],
                  g_num_units=params['gen_g_num_units'],
                  g_num_layers=params['gen_g_num_layers'],
                  aux_size=params['aux_size'],
                  varnames=params['variables'],
                  aux_ou=params['aux_ou'],
                  dawn_dusk_sampler=dawn_dusk_sampler).to(device)
    D = DiscriminatorSimple(data_size=state_size,
                            time_size=params['time_series_length'],
                            num_layers=params['dis_num_layers'],
                            num_units=params['dis_num_units']).to(device)

    G.load_state_dict(torch.load(f'saved_models/sde/sde_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
    D.load_state_dict(torch.load(f'saved_models/sde/sde_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    samples = G.sample(n_samples)
    sde_samples = pd.DataFrame(np.vstack(transformer.inverse_transform(samples).detach().cpu().numpy()), columns=params['variables'])
    sde_samples.to_csv(f"ercot_samples_sde.csv", index=False)


if __name__ == "__main__":
    # fire.Fire(main)
    make_plots()
