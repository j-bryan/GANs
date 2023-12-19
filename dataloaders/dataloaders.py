from typing import Union, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, segment_size: int, preprocessor: Pipeline = None) -> None:
        data = data.values.astype(np.float32)
        data = np.atleast_2d(data)
        if data.shape[0] == 1:  # if only one variable, transpose the data to make it columnar
            data = data.T

        if preprocessor is None:
            preprocessor = Pipeline([('identity', 'passthrough')])
        self.pipe = preprocessor
        self.data = self.pipe.fit_transform(data).T
        self.segment_size = segment_size

    def __len__(self) -> int:
        return len(self.data[0]) // self.segment_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.Tensor(self.data[:, idx * self.segment_size:(idx + 1) * self.segment_size])

    def get_preprocessor(self) -> Pipeline:
        return self.pipe


def _time_features(vardata):
    pass


def _load_data(iso: str, varname: Union[list, str], segment_size: int, preprocessor: Pipeline) -> TimeSeriesDataset:
    if isinstance(varname, str):
        varname = [varname]
    df = pd.read_csv(f'dataloaders/data/{iso.lower()}.csv')
    dataset = TimeSeriesDataset(df[varname], segment_size, preprocessor)
    return dataset


def get_dataloader(iso: str, varname: Union[list, str], segment_size: int = 24, batch_size: int = 32, preprocessor: Pipeline = None) -> Tuple[DataLoader, Pipeline]:
    vardata = _load_data(iso, varname, segment_size, preprocessor)
    dataloader = DataLoader(vardata, batch_size=batch_size, shuffle=True)
    return dataloader, vardata.get_preprocessor()


def get_sde_dataloader(iso: str, varname: Union[list, str], segment_size: int = 24, batch_size: int = 32) -> DataLoader:
    pass
