import torch
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class Preprocessor:
    def __init__(self):
        self.preprocessor = Pipeline([('passthrough', None)])

    def transformed_sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.forward(x).cpu().data.numpy()
        return self.preprocessor.inverse_transform(samples)


class ManualMinMaxScaler(TransformerMixin):
    def __init__(self, input_range, feature_range):
        in_min, in_max = input_range
        out_min, out_max = feature_range

        self.in_min = in_min
        self.out_min = out_min
        self.in_out_scale = (in_max - in_min) / (out_max - out_min)

    def fit(self, X, y=None):
        return self  # no fitting necessary

    def transform(self, X):
        return (X - self.in_min) / self.in_out_scale + self.out_min

    def inverse_transform(self, X):
        return (X - self.out_min) * self.in_out_scale + self.in_min
