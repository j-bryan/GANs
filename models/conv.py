"""
    1-D convolutional GAN as described in Cramer et al.
    This implementation could use some work to make it more flexible for tuning!
"""

import torch
from torch import nn
from models.preprocessing import Preprocessor
from models.layers import FeaturewiseActivation, activations


class Generator(nn.Module, Preprocessor):
    def __init__(self,
                 input_size: int,
                 num_filters: int,
                 num_layers: int,
                 output_size: int,
                 num_vars: int,
                 activation: str = 'lipswish',
                 output_activation: str = 'sigmoid',
                 **kwargs) -> None:
        """
        A convolution-based generator.

        Parameters
        ----------
        input_size : int
            Size of the input latent space.
        num_filters : int
            The number of filters to use in the convolutional layers.
        num_layers : int
            The number of convolutional layers.
        output_size : int
            The size of the output (number of timesteps). All num_vars variables will have this many timesteps.
        num_vars : int
            The number of variables to generate.
        activation : str (default: 'relu')
            The activation function to use for the hidden layers.
        output_activation : str (default: 'identity')
            The activation function to use for the output layer.
        kwargs : dict
            Keyword arguments to pass to the ConvTranspose1D layers.
        """
        super().__init__()
        Preprocessor.__init__(self)
        self.latent_dim = input_size
        self.n_channels = num_filters

        # Intermediate and final activation functions
        interm_activation = activations.get(activation.lower())

        # ConvTranspose1d default parameters
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 1)
        kernel_size = kwargs.get('kernel_size', 3)

        # input will be of shape (batch_size, latent_dim)
        model = [nn.Linear(self.latent_dim, output_size * num_filters),
                 nn.Unflatten(1, (num_filters, output_size))]  # to expand the input from latent_dim to latent_dim*n_channels

        for i in range(num_layers):
            model.append(nn.ConvTranspose1d(num_filters, num_filters, kernel_size=kernel_size, stride=stride, padding=padding))
            model.append(interm_activation())
        model.append(nn.ConvTranspose1d(num_filters, num_vars, kernel_size=kernel_size, stride=stride, padding=padding))

        if isinstance(output_activation, str):
            model.append(activations.get(output_activation.lower())())
        elif isinstance(output_activation, list):
            activations_list = torch.nn.ModuleList([activations.get(act.lower())() for act in output_activation])
            # Need output to be of shape (batch_size, time_steps, num_vars) going into here!
            model.append(FeaturewiseActivation(activations_list, axis=1))

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = output.transpose(1, 2)  # reshape to (batch_size, time_steps, num_vars)
        return output

    def sample_latent(self, num_samples: int = 1) -> torch.Tensor:
        return torch.randn((num_samples, self.latent_dim))

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        return self.forward(self.sample_latent(num_samples))


class Discriminator(nn.Module):
    def __init__(self,
                 num_filters: int,
                 num_layers: int,
                 activation: str = 'lipswish',
                 output_activation: str = 'sigmoid',
                 **kwargs) -> None:
        """
        A convolution-based discriminator. Uses a linear read-out layer and an optional final activation at the output.

        Parameters
        ----------
        num_filters : int
            The number of filters to use in the convolutional layers.
        num_layers : int
            The number of convolutional layers.
        activation : str (default: 'relu')
            The activation function to use for the hidden layers.
        output_activation : str (default: 'identity')
            The activation function to use for the output layer.
        kwargs : dict
            Keyword arguments to pass to the Conv1D layers.
        """
        super().__init__()

        # Default padding and stride values are 1. Combined with a default filter size of 3,
        # this will preserve the input shape.
        kernel_size = kwargs.get('kernel_size', 3)
        padding = kwargs.get('padding', 1)
        stride = kwargs.get('stride', 1)

        # Intermediate and final activation functions
        interm_activation = activations.get(activation.lower())
        final_activation = activations.get(output_activation.lower())

        # LazyConv1d as a first layer gives flexibility in the input shape
        _model = [nn.LazyConv1d(num_filters, kernel_size=kernel_size, stride=stride, padding=padding), interm_activation()]
        # Add the rest of the convolutional layers, all with the same parameters
        # FIXME it would be nice to have a way to specify the number of filters for each layer rather than using the same number for all layers
        for _ in range(num_layers - 1):
            _model.append(nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, stride=stride, padding=padding))
            _model.append(interm_activation())
        # Flatten the output of the convolutional layers so it can be fed into the linear read-out layer
        _model.append(nn.Flatten())
        # Linear read-out layer. Using a LazyLinear layer here since it's unsure what the input shape will be.
        # FIXME add a way to add a full MLP after the convolutional layers rather than just a single linear layer
        _model.append(nn.LazyLinear(1))
        _model.append(final_activation())

        self.model = nn.Sequential(*_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model input x has shape (batch_size, num_vars, num_time_steps)
        return self.model(x)
