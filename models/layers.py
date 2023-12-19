import torch


class LipSwish(torch.nn.Module):
    """
        Swish (silu) activation function, normalized to be 1-Lipschitz. Equal to x * sigmoid(x) / 1.1.
        LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.

        Reference:
        Chen, R. T., Behrmann, J., Duvenaud, D. K., & Jacobsen, J. H. (2019). Residual flows for
        invertible generative modeling. Advances in Neural Information Processing Systems, 32.
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    """
        A simple multi-layer perceptron with LipSwish intermediate activations and a final (optional) tanh
        activation at the output layer.
    """
    activations = {
        'relu': torch.nn.ReLU,
        'leaky_relu': torch.nn.LeakyReLU,
        'lipswish': LipSwish,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid,
        'identity': torch.nn.Identity
    }

    def __init__(self, in_size, out_size, mlp_size, num_layers,
                 activation='lipswish', activation_kwargs={},
                 final_activation='identity', final_activation_kwargs={}):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 self.activations.get(activation.lower())(**activation_kwargs)]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activations.get(activation.lower())(**activation_kwargs))
        model.append(torch.nn.Linear(mlp_size, out_size))
        model.append(self.activations.get(final_activation.lower())(**final_activation_kwargs))
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
