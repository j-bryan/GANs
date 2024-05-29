"""
We want the solar drift function to be of the form
    f(t, x) = a(t, x) * H(t - t_dawn) * H(t_dusk - t) * cos(pi (t - t_dawn) / (t_dusk - t_dawn))
The a(t, x) function is parameterized by a neural network. H is the Heaviside step function. The dawn
and dusk times are sampled from the solar data.

The solar diffusion function is of the form
    g(t, x) = b(t, x) * H(t - t_dawn) * H(t_dusk - t) * x * (1 - x)

The wind drift function is of the form
    h(t, x) = theta(t, x) * (x - mu(t, x))
This is a simple mean-reverting drift function whose mean mu and diffusion coefficient theta are
functions of t (and x) and are parameterized with neural networks. We implement just the (x - mu(t))
term here, with the other portion of the drift function implemented in the SDE generator class.

The wind diffusion function is of the form
    m(t, x) = d(t, x) * x * (1 - x)
This will ensure that the wind function is always positive
"""

import torch
from models.layers import MLP


class DawnDuskSampler:
    def __init__(self, data: torch.Tensor):
        self.data = data  # shape (*, segment_size) of solar data
        self.segment_length = data.size(1)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.data.size(0), (n,))
        samples = self.data[idx]
        # Find the dawn and dusk times of the samples
        dawn = torch.zeros(n)
        dusk = torch.zeros(n)
        for i in range(n):
            # Find the first time the solar data is non-zero
            is_daytime = (samples[i] > 1e-3).float()
            dawn[i] = torch.argmax(is_daytime)
            # Find the last time the solar data is non-zero
            dusk[i] = self.segment_length - torch.argmax(is_daytime.flip(0))
        return dawn, dusk


class FastHardSigmoid(torch.nn.Module):
    def __init__(self, slope: float = 1.0):
        super().__init__()
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.slope + 0.5, 0, 1)


class SolarMultForcing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = FastHardSigmoid(slope=1.0)

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        t_dawn = kwargs['t_dawn']
        t_dusk = kwargs['t_dusk']

        # During the day, the drift term is multiplied by a cosine function that is at its maximum
        # absolute values at dawn and dusk.
        drift = torch.cos(torch.pi * (t - t_dawn) / (t_dusk - t_dawn)) \
                * self.threshold(t - t_dawn) * self.threshold(t_dusk - t)

        diffusion = self.threshold(t - t_dawn) * self.threshold(t_dusk - t)

        return drift, diffusion


class SolarAddForcing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = FastHardSigmoid(slope=1.0)
        self.nighttime_mean_reversion = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        t_dawn = kwargs['t_dawn']
        t_dusk = kwargs['t_dusk']
        drift = self.nighttime_mean_reversion * x * self.threshold(t_dawn - t) * self.threshold(t - t_dusk)
        diffusion = torch.zeros_like(x)
        return drift, diffusion


class WindMultForcing(torch.nn.Module):
    def __init__(self, in_size:    int =  2,
                       out_size:   int =  1,
                       mlp_size:   int = 64,
                       num_layers: int =  2,
                       **kwargs):
        super().__init__()
        self.model = MLP(in_size, out_size, mlp_size, num_layers, **kwargs)

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs):
        # drift_term = x - self.model(torch.vstack([t, x]).T).squeeze()
        drift_term = torch.ones_like(x)
        # diffusion_term = x * (1 - x)
        diffusion_term = torch.ones_like(x)
        return drift_term, diffusion_term
