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
                       device: str = "cpu",
                       test_size: float = 0.0,
                       valid_size: float = 0.0) -> DataLoader:
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
    device : str (default: "cpu")
        The device to load the data on.
    test_size : float (default: 0.0)
        The proportion of the data to use as a test set. If 0, all data is used for training. Must
        be between 0 and 1.
    valid_size : float (default: 0.0)
        The proportion of the data to use as a validation set. If 0, no validation set is used. Must
        be between 0 and 1.

    Returns
    -------
    dataloader : DataLoader
        A pytorch DataLoader object.
    """
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    if test_size + valid_size >= 1:
        raise ValueError(f"test_size + valid_size must be less than 1. Got test_size={test_size} and "
                         f"valid_size={valid_size}, so test_size + valid_size = {test_size + valid_size}")

    data = pd.read_csv(f"dataloaders/data/{iso.lower()}_eia.csv")[varname]
    if "SOLAR" in varname:
        # Some solar data has non-zero values at night
        solar = data["SOLAR"].values.reshape(-1, 24)
        night_hours = np.where(solar.mean(axis=0) < 1e-3)[0]
        solar[:, night_hours] = 0
        data["SOLAR"] = solar.ravel()
    data = data.to_numpy()
    data = torch.Tensor(data.reshape(-1, segment_size, len(varname)))

    idx = np.arange(len(data))
    np.random.shuffle(idx)
    i_train_split = int(len(data) * (1 - test_size - valid_size))
    i_valid_split = int(len(data) * (1 - test_size))
    train_data = data[idx[:i_train_split]]
    valid_data = data[idx[i_train_split:i_valid_split]] if valid_size > 0 else None
    test_data = data[idx[i_valid_split:]] if test_size > 0 else None

    # The hard-coded mean and std are for ERCOT 2022 data. The data in ercot_eia.csv is already scaled
    # using a StandardScaler by year. These values will invert that scaling and produce 2022 levels.
    load_transformer = StandardScaler(mean=49073.591773469176, std=10502.989133706267)
    if "TOTALLOAD" in varname:
        transformer = InvertibleColumnTransformer(
            transformers={
                "TOTALLOAD": load_transformer,
            },
            columns=varname
        )
        Xt = transformer.fit_transform(train_data)
    else:
        Xt = train_data
        transformer = Passthrough()

    train_dataloader = torch.utils.data.DataLoader(Xt.to(device), batch_size=batch_size, shuffle=True)

    if test_data is not None:
        test_data = transformer.transform(test_data)
        test_dataloader = torch.utils.data.DataLoader(test_data.to(device), batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None

    if valid_data is not None:
        valid_data = transformer.transform(valid_data)
        valid_dataloader = torch.utils.data.DataLoader(valid_data.to(device), batch_size=batch_size, shuffle=False)
    else:
        valid_dataloader = None

    return train_dataloader, test_dataloader, valid_dataloader, transformer
