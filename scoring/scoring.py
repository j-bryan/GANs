"""
Calculation and comparison of certain metrics of interest for synthetic histories.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf


def hourly_distributions(synthetic: np.ndarray, historical: np.ndarray) -> None:
    """
    Plots hourly distributions of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Data loader with the historical data
    """
    df = pd.DataFrame(columns=['Hour', 'Variable', 'Dataset', 'Value'])
    for j in range(synthetic.shape[1]):  # loop over variables
        for i in range(synthetic.shape[2]):  # loop over timesteps
            df_synth = pd.DataFrame({'Hour': i, 'Variable': j, 'Dataset': 'Synthetic', 'Value': synthetic[:, j, i]})
            df_hist = pd.DataFrame({'Hour': i, 'Variable': j, 'Dataset': 'Historical', 'Value': historical[:, j, i]})
            df = pd.concat([df, df_synth, df_hist], ignore_index=True)
    df = df.reset_index(drop=True)

    fig = px.box(df, x='Hour', y='Value', color='Dataset', facet_col='Variable', facet_col_wrap=2)
    fig.show()


def autocorrelation(synthetic: np.ndarray, historical: np.ndarray, varnames: list = None) -> None:
    """
    Plots the autocorrelation of the synthetic and historical data.

    Parameters
    ----------
    synthetic : np.ndarray
        Synthetic data, shape (num_samples, num_variables, num_timesteps)
    historical : np.ndarray
        Data loader with the historical data
    varnames : list, optional
        List of variable names
    """
    fig, ax = plt.subplots(nrows=synthetic.shape[1], ncols=1, figsize=(10, 10/3 * synthetic.shape[1]))
    for ivar in range(synthetic.shape[1]):
        synth = synthetic[:, ivar, :].flatten()
        hist = historical[:, ivar, :].flatten()

        plot_acf(synth, ax=ax[ivar], label='Synthetic')
        plot_acf(hist, ax=ax[ivar], label=f'Historical')
        ax[ivar].set_ylabel(varnames[ivar] if varnames is not None else f'Variable {ivar}')

        # drop every other item in the legend
        handles, labels = ax[ivar].get_legend_handles_labels()
        ax[ivar].legend(handles[1::2], labels[1::2])

    plt.show()
