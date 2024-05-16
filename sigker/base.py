import torch
import torchcde
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class LipSwish(torch.nn.Module):
    """
    LipSwish activation to control Lipschitz constant of MLP output
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class ScaledTanh(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        return self.scale * torch.nn.Tanh()(x)


class MLP(torch.nn.Module):
    """
    Standard multi-layer perceptron
    """

    def __init__(self, in_size, out_size, mlp_size, num_layers, activation="LipSwish", tanh=True, tscale=1):
        """
        Initialisation of perceptron

        :param in_size:     Size of data input
        :param out_size:    Output data size
        :param mlp_size:    Number of neurons in each hidden layer
        :param num_layers:  Number of hidden layers
        :param activation:  Activation function to use between layers.
        :param tanh:        Whether to apply tanh activation to final linear layer
        :param tscale:      Custom scaler to tanh layer
        """
        super().__init__()

        if activation != "LipSwish":
            self.activation = getattr(torch.nn, activation)
        else:
            self.activation = LipSwish

        model = [torch.nn.Linear(in_size, mlp_size), self.activation()]

        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activation())

        model.append(torch.nn.Linear(mlp_size, out_size))

        if tanh:
            model.append(ScaledTanh(tscale))

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
