import numpy as np
import torch
from models.layers import MLP
from models.preprocessing import Preprocessor


class Generator(torch.nn.Module, Preprocessor):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 num_vars: int = 1) -> None:
        """
        A simple feedforward generator.

        Parameters
        ----------
        input_size : int
            The size of the latent space.
        hidden_size : int
            The size of the hidden layers.
        num_layers : int
            The number of hidden layers.
        output_size : int
            The size of the output (number of timesteps). All num_vars variables will have this many timesteps.
        num_vars : int (default: 1)
            The number of variables to generate.
        """
        super().__init__()
        Preprocessor.__init__(self)
        self.latent_dim = input_size
        mlp = MLP(input_size,
                  output_size * num_vars,  # ensures the tensor can be reshaped into (num_vars, output_size)
                  hidden_size,
                  num_layers,
                  activation='lipswish',
                  final_activation='sigmoid')
        unflatten = torch.nn.Unflatten(1, (num_vars, output_size))
        self.model = torch.nn.Sequential(mlp, unflatten)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def sample_latent(self, num_samples: int = 1) -> torch.Tensor:
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_vars: int = 1) -> None:
        """
        A simple feedforward discriminator.

        Parameters
        ----------
        input_size : int
            The size of the input (number of timesteps). All num_vars variables will have this many timesteps.
        hidden_size : int
            The size of the hidden layers.
        num_layers : int
            The number of hidden layers.
        num_vars : int (default: 1)
            The number of variables to generate.
        """
        super().__init__()

        flatten = torch.nn.Flatten()  # dimension agnostic, will work for any number of variables
        mlp = MLP(input_size * num_vars,
                  1,
                  hidden_size,
                  num_layers,
                  activation='lipswish',
                  final_activation='tanh')
        self.model = torch.nn.Sequential(flatten, mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
