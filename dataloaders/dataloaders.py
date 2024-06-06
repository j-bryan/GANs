from typing import Union, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from dataloaders.preprocessing import StandardScaler, FunctionTransformer, InvertibleColumnTransformer

from dataloaders.data.meta import meta


class Passthrough:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 segment_size: int,
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
                       batch_size: int = 32,
                       device: str = "cpu") -> DataLoader:
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
    data = pd.read_csv(f"dataloaders/data/{iso.lower()}.csv")[varname]
    if "SOLAR" in varname:
        # Some solar data has non-zero values at night
        solar = data["SOLAR"].values.reshape(-1, 24)
        night_hours = np.where(solar.mean(axis=0) < 1e-3)[0]
        solar[:, night_hours] = 0
        data["SOLAR"] = solar.ravel()
    data = data.to_numpy()
    data = torch.Tensor(data.reshape(-1, segment_size, len(varname)))

    load_transformer = StandardScaler()
    if "TOTALLOAD" in varname:
        transformer = InvertibleColumnTransformer(
            transformers={
                "TOTALLOAD": load_transformer,
            },
            columns=varname
        )
        Xt = transformer.fit_transform(data)
    else:
        Xt = data
        transformer = Passthrough()

    dataloader = torch.utils.data.DataLoader(Xt.to(device), batch_size=batch_size, shuffle=True)

    return dataloader, transformer
