import torch


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


class MLP(torch.nn.Module):
    """
        A simple multi-layer perceptron with intermediate activations and a final optional
        activation at the output layer.
    """
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 mlp_size: int,
                 num_layers: int,
                 activation: str = 'lipswish',
                 activation_kwargs: dict = None,
                 final_activation: str = 'identity',
                 final_activation_kwargs: dict = None) -> None:
        """
        Class constructor.

        Parameters
        ----------
        in_size : int
            The size of the input.
        out_size : int
            The size of the output.
        mlp_size : int
            The size of the hidden layers.
        num_layers : int
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

        model = [torch.nn.Linear(in_size, mlp_size),
                 activations.get(activation.lower())(**activation_kwargs)]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(activations.get(activation.lower())(**activation_kwargs))
        model.append(torch.nn.Linear(mlp_size, out_size))
        model.append(activations.get(final_activation.lower())(**final_activation_kwargs))
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
        'sigmoid': torch.nn.Sigmoid,
        'identity': torch.nn.Identity
    }
