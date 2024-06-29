import numpy as np
import torch


class StandardScaler:
    """
    Implements a scikit-learn style StandardScaler that is compatible with batched pytorch tensors
    without needing to convert to numpy arrays.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self._fitted = mean is not None and std is not None

    def fit(self, X):
        if self._fitted:
            return self
        self.mean = X.mean()
        self.std = X.std()
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def __call__(self, X):
        return self.transform(X)


class FunctionTransformer:
    """
    Implements a scikit-learn style FunctionTransformer that is compatible with batched pytorch tensors
    without needing to convert to numpy arrays.
    """
    def __init__(self, func=None, inverse_func=None):
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X):
        return self

    def transform(self, X):
        if self.func is None:
            return X
        return self.func(X)

    def fit_transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return X
        return self.inverse_func(X)

    def __call__(self, X):
        return self.transform(X)


class Pipeline:
    """
    Implements a scikit-learn style pipeline that is compatible with batched pytorch tensors
    without needing to convert to numpy arrays.
    """
    def __init__(self, *args):
        self.steps = args

    def fit(self, X):
        for step in self.steps:
            X = step.fit_transform(X)

    def transform(self, X):
        for step in self.steps:
            X = step(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        for step in self.steps[::-1]:
            X = step.inverse_transform(X)
        return X

    def __call__(self, X):
        return self.transform(X)


def make_pipeline(*args):
    """
    Make a pipeline of functions that can be applied to a batched pytorch tensor.
    """
    return Pipeline(*args)


class InvertibleColumnTransformer:
    def __init__(self, transformers: dict, columns: list[str]):
        """
        Transformer that applies a set of transformations to a set of columns. The name of all columns
        must be provided, but only the ones that are in the transformers dictionary will be transformed.
        All others will be passed through unchanged.

        :param transformers: Dictionary of scikit-learn style transformers, where the keys are a
                             subset of the columns.
        :param columns: List of columns to apply the transformers to. The order of the names must
                        correspond to the order of the columns in the input data.
        """
        self.transformers = transformers
        self.columns = columns

    def fit(self, X: np.ndarray | torch.Tensor, y=None):
        for i, column in enumerate(self.columns):
            if column not in self.transformers:
                continue
            self.transformers[column].fit(X[..., i].reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(X, torch.Tensor)
        Xt = X.detach().clone() if is_tensor else X.copy()
        for i, column in enumerate(self.columns):
            if column not in self.transformers:
                continue
            xt_col = self.transformers[column].transform(X[..., i].reshape(-1, 1))
            Xt[..., i] = xt_col.reshape(Xt[..., i].shape)
        return Xt

    def fit_transform(self, X: np.ndarray | torch.Tensor, y=None) -> np.ndarray | torch.Tensor:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(X, torch.Tensor)
        Xt = X.detach().clone() if is_tensor else X.copy()
        for i, column in enumerate(self.columns):
            if column not in self.transformers:
                continue
            xt_col = self.transformers[column].inverse_transform(X[..., i].reshape(-1, 1))
            Xt[..., i] = xt_col.reshape(Xt[..., i].shape)
        return Xt
