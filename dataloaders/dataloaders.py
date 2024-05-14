from typing import Union, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline

from dataloaders.data.meta import meta


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 segment_size: int,
                 preprocessor: Pipeline = None,
                 transpose: bool = False) -> None:
        data = data.values.astype(np.float32)
        data = np.atleast_2d(data)
        if data.shape[0] == 1:  # if only one variable, transpose the data to make it columnar
            data = data.T

        if preprocessor is None:
            preprocessor = Pipeline([('identity', 'passthrough')])
        self.pipe = preprocessor
        self.data = self.pipe.fit_transform(data).T

        self.segment_size = segment_size
        self.transpose = transpose

    def __len__(self) -> int:
        return len(self.data[0]) // self.segment_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = torch.Tensor(self.data[:, idx * self.segment_size:(idx + 1) * self.segment_size])
        if self.transpose:
            return item.transpose(0, 1)
        else:
            return item

    def get_preprocessor(self) -> Pipeline:
        return self.pipe


def _load_data(iso: str,
               varname: Union[list[str], str],
               segment_size: int,
               preprocessor: Pipeline) -> TimeSeriesDataset:
    if isinstance(varname, str):
        varname = [varname]
    df = pd.read_csv(f'dataloaders/data/{iso.lower()}.csv')
    dataset = TimeSeriesDataset(df[varname], segment_size, preprocessor)
    return dataset


def get_dataloader(iso: str,
                   varname: Union[list[str], str],
                   segment_size: int = 24,
                   batch_size: int = 32,
                   preprocessor: Pipeline = None) -> Tuple[DataLoader, Pipeline]:
    """
    Creates a pytorch dataloader from the specified data.

    Parameters
    ----------
    iso : str
        The ISO to load data from.
    varname : Union[list, str]
        The variable(s) to load.
    segment_size : int (default: 24)
        The number of time steps to include in each sample.
    batch_size : int (default: 32)
        The batch size.
    preprocessor : Pipeline (default: None)
        A sklearn pipeline to preprocess the data.

    Returns
    -------
    dataloader : DataLoader
        A pytorch DataLoader object.
    """
    vardata = _load_data(iso, varname, segment_size, preprocessor)
    dataloader = DataLoader(vardata, batch_size=batch_size, shuffle=True)
    return dataloader, vardata.get_preprocessor()


def get_sde_dataloader(iso: str,
                       varname: Union[list[str], str],
                       segment_size: int = 24,
                       time_features: Union[list[str], str] = None,
                       batch_size: int = 32,
                       preprocessor: Pipeline = None) -> DataLoader:
    """
    Creates a pytorch dataloader from the specified data. Differs from the standard dataloader in
    that it interpolates the data for use with continuous-time models (e.g. SDEs).

    Parameters
    ----------
    iso : str
        The ISO to load data from.
    varname : Union[list, str]
        The variable(s) to load.
    segment_size : int (default: 24)
        The number of time steps to include in each sample.
    batch_size : int (default: 32)
        The batch size.
    preprocessor : Pipeline (default: None)
        A sklearn pipeline to preprocess the data.

    Returns
    -------
    dataloader : DataLoader
        A pytorch DataLoader object.
    """
    # lazy imports
    import torchcde

    vardata = _load_data(iso, varname, segment_size, preprocessor)
    vardata.transpose = True

    if time_features is not None:
        # hard-coded meta data for more info on the ISO data found in dataloaders/data/meta.py
        start_dt = meta[iso]['start_dt']
        end_dt = meta[iso]['end_dt']
        freq = meta[iso]['freq']
        date_range = pd.date_range(start_dt, end_dt, freq=freq)

        if isinstance(time_features, str):
            time_features = [time_features]

        for feat in time_features[::-1]:
            if feat == 'HOD':
                feat_data = np.array(date_range.hour)
            elif feat == 'DOW':
                feat_data = np.array(date_range.dayofweek)
            elif feat == 'MOY':
                feat_data = np.array(date_range.month)
            else:
                raise ValueError(f'Invalid time feature {feat}. Please choose from "HOD", "DOW", or "MOY".')
            vardata.data = np.vstack([feat_data, vardata.data])

    # TODO: need to interpolate the data to get a continuous-time representation?
    # data = np.stack([hours, wind], axis=1)
    # data = np.array(np.split(data, dataset_size))
    # ys = torch.tensor(data, dtype=torch.float32, device=device)
    # ts = torch.linspace(0, t_size - 1, t_size, device=device)
    # data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    # ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    # dataset = torch.utils.data.TensorDataset(ys_coeffs)

    dataloader = torch.utils.data.DataLoader(vardata, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dataloader, preprocessor
