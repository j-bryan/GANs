"""
Calculation and comparison of certain metrics of interest for synthetic histories.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_ccf

from itertools import combinations
from scipy.stats import wasserstein_distance


def distributions(synthetic: np.ndarray, historical: np.ndarray, varnames: list[str]) -> None:
    """
    Plots distributions of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Historical data, shape (num_samples, num_variables, num_timesteps)
    varnames : list
        List of variable names
    """
    df = None
    for j in range(synthetic.shape[1]):  # loop over variables
        df_synth = pd.DataFrame({'Variable': varnames[j], 'Dataset': 'Synthetic',  'Value':  synthetic[:, j, :].flatten()})
        df_hist  = pd.DataFrame({'Variable': varnames[j], 'Dataset': 'Historical', 'Value': historical[:, j, :].flatten()})
        if df is None:
            df = pd.concat([df_synth, df_hist], ignore_index=True)
        else:
            df = pd.concat([df, df_synth, df_hist], ignore_index=True)
    df = df.reset_index(drop=True)

    for vn, df in df.groupby('Variable'):
        print(f'Wasserstein distance for {vn}: {wasserstein_distance(df[df.Dataset == "Synthetic"].Value, df[df.Dataset == "Historical"].Value)}')

    fig = px.histogram(df, x='Value', facet_col='Variable', color='Dataset', barmode='overlay')
    fig.show()


def hourly_distributions(synthetic: np.ndarray, historical: np.ndarray, varnames: list[str]) -> None:
    """
    Plots hourly distributions of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Historical data, shape (num_samples, num_variables, num_timesteps)
    varnames : list
        List of variable names
    """
    df = None
    for j in range(synthetic.shape[1]):  # loop over variables
        for i in range(synthetic.shape[2]):  # loop over timesteps
            df_synth = pd.DataFrame({'Hour': i, 'Variable': varnames[j], 'Dataset': 'Synthetic',  'Value':  synthetic[:, j, i]})
            df_hist  = pd.DataFrame({'Hour': i, 'Variable': varnames[j], 'Dataset': 'Historical', 'Value': historical[:, j, i]})
            if df is None:
                df = pd.concat([df_synth, df_hist], ignore_index=True)
            else:
                df = pd.concat([df, df_synth, df_hist], ignore_index=True)
    df = df.reset_index(drop=True)

    fig = px.box(df, x='Hour', y='Value', color='Dataset', facet_col='Variable', facet_col_wrap=2)
    fig.show()


def autocorrelation(synthetic: np.ndarray, historical: np.ndarray, varnames: list, lags: int = None) -> None:
    """
    Plots the autocorrelation of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Historical data, shape (num_samples, num_variables, num_timesteps)
    varnames : list[str]
        List of variable names
    lags : int, optional
        Number of lags to include in the autocorrelation function. Default value of None will use the
        default value calculated by statsmodels.
    """
    fig, ax = plt.subplots(nrows=synthetic.shape[1], ncols=1, figsize=(10, 10/3 * synthetic.shape[1]))
    for ivar in range(synthetic.shape[1]):
        synth = synthetic[:, ivar, :].flatten()
        hist = historical[:, ivar, :].flatten()

        plot_acf(synth, lags=lags, ax=ax[ivar], label='Synthetic')
        plot_acf(hist,  lags=lags, ax=ax[ivar], label='Historical')
        ax[ivar].set_ylabel(varnames[ivar] if varnames is not None else f'Variable {ivar}')

        # drop every other item in the legend
        handles, labels = ax[ivar].get_legend_handles_labels()
        ax[ivar].legend(handles[1::2], labels[1::2])

    plt.show()


def cross_correlation(synthetic: np.ndarray, historical: np.ndarray, varnames: list, lags: int = None) -> None:
    """
    Plots the cross-correlation of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Historical data, shape (num_samples, num_variables, num_timesteps)
    varnames : list[str]
        List of variable names
    lags : int, optional
        Number of lags to include in the autocorrelation function. Default value of None will use the
        default value calculated by statsmodels.
    """
    var_combs = list(combinations(range(synthetic.shape[1]), 2))  # all combinations of two variables
    fig, ax = plt.subplots(nrows=len(var_combs), ncols=1, figsize=(10, 10/3 * len(var_combs)))
    if len(var_combs) == 1:
        ax = [ax]

    for i, (ivar1, ivar2) in enumerate(var_combs):
        synth1 = synthetic[:, ivar1, :].flatten()
        synth2 = synthetic[:, ivar2, :].flatten()
        hist1 = historical[:, ivar1, :].flatten()
        hist2 = historical[:, ivar2, :].flatten()

        lag_values = np.arange(-lags, lags+1)  # FIXME: This shouldn't be hard-coded
        plot_ccf(synth1, synth2, lags=lag_values, ax=ax[i], label='Synthetic')
        plot_ccf( hist1,  hist2, lags=lag_values, ax=ax[i], label='Historical')
        ax[i].set_xlabel('Lag')
        ax[i].set_ylabel(f'{varnames[ivar1]} x {varnames[ivar2]}')

        # drop every other item in the legend
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].legend(handles[1::2], labels[1::2])

    plt.show()
