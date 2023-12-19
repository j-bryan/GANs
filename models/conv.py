"""
    1-D convolutional GAN as described in Cramer et al.
    This implementation could use some work to make it more flexible for tuning!
"""

import torch
from torch import nn


class Generator(nn.Module):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'leakyrelu': nn.LeakyReLU,
        'lipswish': nn.SiLU
    }
    def __init__(self, input_size, num_filters, num_layers, output_size, activation='relu', output_activation='sigmoid'):
        super().__init__()
        self.latent_dim = input_size
        self.n_channels = num_filters
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * num_filters),  # to expand the input from latent_dim to latent_dim*n_channels
            self.activations.get(activation)(),
            nn.Unflatten(1, (num_filters, self.latent_dim)),
            self.activations.get(activation)(),
            nn.ConvTranspose1d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            self.activations.get(activation)(),
            nn.ConvTranspose1d(num_filters, 1, kernel_size=3, stride=1, padding=1)  # FIXME can't change output size with current parameterization
        )

    def forward(self, x):
        return self.model(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 12, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(12, 4, 3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(4 * 24, 1)  # n_channels * generated_length
            # swap Linear with MLP?
        )

    def forward(self, x):
        return self.model(x)
