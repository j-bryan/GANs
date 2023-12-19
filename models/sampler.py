import numpy as np
from torch import nn
from sklearn.pipeline import Pipeline


class ModelSampler:
    def __init__(self, model: nn.Module, preprocessor: Pipeline):
        self.model = model
        self.preprocessor = preprocessor

    def sample_model(self, num_samples: int = 1) -> np.ndarray:
        """ Samples the generator model and transforms the output back into the original space. """
        latent = self.model.sample_latent(num_samples)
        samples = self.model(latent)
        samples_origspace = self.preprocessor.inverse_transform(samples.numpy(force=True))
        return samples_origspace
