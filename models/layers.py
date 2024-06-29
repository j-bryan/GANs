import torch
from dataclasses import dataclass


@dataclass
class FFNNConfig:
    in_size: int
    num_units: int
    num_hidden_layers: int
    out_size: int
    activation: str = "relu"
    final_activation: str = "identity"

    def to_dict(self, prefix: str = ""):
        # prepend the prefix to the keys
        if prefix == "":
            return self.__dict__
        return {prefix + "_" + k: v for k, v in self.__dict__.items()}


class LipSwish(torch.nn.Module):
    """
        Swish (silu) activation function, normalized to be 1-Lipschitz. Equal to x * sigmoid(x) / 1.1.
        LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.

        Reference:
        Chen, R. T., Behrmann, J., Duvenaud, D. K., & Jacobsen, J. H. (2019). Residual flows for
        invertible generative modeling. Advances in Neural Information Processing Systems, 32.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.909 * torch.nn.functional.silu(x)


class FeaturewiseActivation(torch.nn.Module):
    """
        A module that applies a different activation function to each feature in the input tensor.
    """
    def __init__(self, activations: torch.nn.ModuleList, axis: int = -1):
        super().__init__()
        self.activations = torch.nn.ModuleList(activations)
        self._axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._axis == -1 or self._axis == 2:
            return  torch.stack([act(x[..., i]) for i, act in enumerate(self.activations)], dim=-1)
        elif self._axis == 1:
            return torch.stack([act(x[:, i]) for i, act in enumerate(self.activations)], dim=1)


class FFNN(torch.nn.Module):
    """
        A simple multi-layer perceptron with intermediate activations and a final optional
        activation at the output layer.
    """
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 num_units: int,
                 num_hidden_layers: int,
                 activation: str = 'relu',
                 activation_kwargs: dict = None,
                 final_activation: str | list[str] = 'identity',
                 final_activation_kwargs: dict = None) -> None:
        """
        Class constructor.

        Parameters
        ----------
        in_size : int
            The size of the input.
        out_size : int
            The size of the output.
        num_units : int
            The size of the hidden layers.
        num_hidden_layers : int
            The number of hidden layers.
        activation : str (default: 'lipswish')
            The activation function to use for the hidden layers.
        activation_kwargs : dict (default: None)
            Keyword arguments to pass to the activation function.
        final_activation : str (default: 'identity')
            The activation function to use for the output layer.
        final_activation_kwargs : dict (default: None)
            Keyword arguments to pass to the activation function.
        """
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}
        if final_activation_kwargs is None:
            final_activation_kwargs = {}

        # Handle the case of a single layer
        if num_hidden_layers == 0:
            model = [torch.nn.Linear(in_size, out_size)]
            if isinstance(final_activation, str):
                model.append(activations.get(final_activation.lower())(**final_activation_kwargs))
            else:  # final_activation is a list of activations
                final_activations = torch.nn.ModuleList([activations.get(act.lower())(**final_activation_kwargs) for act in final_activation])
                model.append(FeaturewiseActivation(final_activations))
            self._model = torch.nn.Sequential(*model)
            return

        model = [torch.nn.Linear(in_size, num_units),
                 activations.get(activation.lower())(**activation_kwargs)]
        for _ in range(num_hidden_layers):
            model.append(torch.nn.Linear(num_units, num_units))
            model.append(activations.get(activation.lower())(**activation_kwargs))
        model.append(torch.nn.Linear(num_units, out_size))

        if isinstance(final_activation, str):
            model.append(activations.get(final_activation.lower())(**final_activation_kwargs))
        else:  # final_activation is a list of activations
            final_activations = torch.nn.ModuleList([activations.get(act.lower())(**final_activation_kwargs) for act in final_activation])
            model.append(FeaturewiseActivation(final_activations))

        self._model = torch.nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class Conv1D(torch.nn.Module):
    """
        A 1D convolutional network with intermediate activations, a linear read-out layer, and an
        optional final activation at the output layer.
    """
    def __init__(self,
                 input_size: int,
                 num_filters: int,
                 filter_size: int,
                 num_layers: int,
                 output_size: int,
                 activation: str = 'lipswish',
                 final_activation: str = 'identity',
                 **kwargs) -> None:
        super().__init__()
        """
        Class constructor.

        Parameters
        ----------
        input_size : int
            The size of the input.
        num_filters : int
            The number of filters to use in the convolutional layers.
        filter_size : int
            The size of the filters.
        num_layers : int
            The number of convolutional layers.
        output_size : int
            The size of the output.
        activation : str (default: 'lipswish')
            The activation function to use for the hidden layers.
        final_activation : str (default: 'identity')
            The activation function to use for the output layer.
        """


# map strings to activation function classes for easier construction
activations = {
        'relu': torch.nn.ReLU,
        'leakyrelu': torch.nn.LeakyReLU,
        'lipswish': LipSwish,
        'tanh': torch.nn.Tanh,
        'hardtanh': torch.nn.Hardtanh,
        'sigmoid': torch.nn.Sigmoid,
        'hardsigmoid': torch.nn.Hardsigmoid,
        'identity': torch.nn.Identity
    }
