""" Implements scalers for preprocessing that are compatible with pytorch Tensor objects. """
from Torch import Tensor


class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        self.feature_range = feature_range

    def fit(self, X: Tensor) -> None:
        self.min = X.min()
        self.max = X.max()
        return self

    def transform(self, X: Tensor) -> Tensor:
        return (X - self.min) / (self.max - self.min) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
