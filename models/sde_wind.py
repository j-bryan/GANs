import torch
import torchsde

from models.layers import FFNN
from models.sde_no_embed import InitialCondition


class GeneratorFunc(torch.nn.Module):
    """
        The generator SDE. The drift and diffusion functions as MLPs.
    """
    sde_type = 'stratonovich'
    noise_type = 'diagonal'

    def __init__(self,
                 state_size: int,
                 mlp_size: int,
                 num_layers: int,
                 varnames: list[str] = None) -> None:
        """
        Constructor for the SDE

        Parameters
        ----------
        state_size : int
            The dimensionality of the state (number of variables).
        mlp_size : int
            The size of the hidden layers in the MLPs.
        num_layers : int
            The number of hidden layers in the MLPs.
        varnames : list[str]
            The names of the variables in the data, used to applying exogenous forcing functions to
            the drift and diffusion functions of the SDE by variable.
        add_forcing : dict
            A dictionary of forcing functions (implemented as torch.nn.Modules) indexed by variable name.
            These will be applied additively to the drift and diffusion functions of the SDE.
        mult_forcing : dict
            A dictionary of forcing functions (implemented as torch.nn.Modules) indexed by variable name.
            These will be applied multiplicatively to the drift and diffusion functions of the SDE.
        """
        super().__init__()
        # General drift and diffusion functions modeled with MLPs.
        self.varnames = varnames

        # Wind series parameters
        self._theta_x = torch.nn.Parameter(torch.rand(1))
        self._mu_x    = torch.nn.Parameter(torch.tensor(0.5))
        self._sigma_x = torch.nn.Parameter(torch.rand(1))
        self._alpha_x = torch.nn.Parameter(torch.rand(1))
        self._phi_x   = torch.nn.Parameter(torch.rand(1))
        self._beta1_x = torch.nn.Parameter(torch.rand(1))
        self._beta2_x = torch.nn.Parameter(torch.rand(1))
        # Auxiliary series parameters
        self._theta_u = torch.nn.Parameter(torch.rand(1))
        self._mu_u    = torch.nn.Parameter(torch.tensor(0.0))
        self._sigma_u = torch.nn.Parameter(torch.rand(1))

        # self._drift = MLP(
        #     in_size=1 + state_size,  # +1 for time dimension
        #     out_size=state_size,
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='identity'
        # )
        # self._diffusion = MLP(
        #     in_size=1 + state_size,
        #     out_size=state_size,
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='identity'
        # )

    def f_and_g(self,
                t: torch.Tensor,
                x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used by torchsde, not a pytorch convention. Returns drift and diffusion
        values.

        Parameters
        ----------
        t : torch.Tensor
            The time.
        x : torch.Tensor
            The current state of the SDE.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The drift and diffusion terms of the SDE.
        """
        f = torch.zeros_like(x)
        g = torch.zeros_like(x)

        X = x[:, 0]
        U = x[:, 1]

        sig = torch.nn.functional.sigmoid

        # f[:, 0] = (self._alpha_x * torch.sin(2 * torch.pi / 24 * t + self._phi_x) + U) * (1 - torch.exp(-X)) \
        #             + self._theta_x ** 2 * (sig(self._mu_x) - X)
        # g[:, 0] = self._sigma_x ** 2 * X ** (self._beta1_x ** 2) * (1 - X) ** (self._beta2_x ** 2)

        # f[:, 1] = self._theta_u ** 2 * (self._mu_u - U)
        # g[:, 0] = self._sigma_u ** 2

        relu = torch.nn.functional.relu

        f[:, 0] = (self._alpha_x * torch.sin(2 * torch.pi / 24 * t + self._phi_x) + U) * (1 - torch.exp(-X)) * (1 - torch.exp(1 - X)) \
                    + relu(self._theta_x) * (sig(self._mu_x) - X)
        g[:, 0] = relu(self._sigma_x) * X ** relu(self._beta1_x) * (1 - X) ** relu(self._beta2_x)

        f[:, 1] = relu(self._theta_u) * (self._mu_u - U)
        g[:, 0] = relu(self._sigma_u)

        # if torch.any(x[:, 0] <= 1e-2) or torch.any(x[:, 0] >= 1 - 1e-2):
        if torch.any(x[:, 0] <= 0) or torch.any(x[:, 0] >= 1):
            bad_idx = torch.logical_or(x[:, 0] <= 0, x[:, 0] >= 1)
            print('Warning: x out of bounds!')
            print("t", t)
            print("x", x[bad_idx])
            print("f", f[bad_idx])
            print("g", g[bad_idx])
            print(X[bad_idx] ** relu(self._beta1_x))
            print((1 - X[bad_idx]) ** relu(self._beta2_x))
            # print(X ** (self._beta1_x ** 2)[bad_idx])
            # print((1 - X) ** (self._beta2_x ** 2)[bad_idx])
            print("X parameters")
            print(f"... theta = {relu(self._theta_x).item()}")
            print(f"... mu    = {sig(self._mu_x).item()}")
            print(f"... sigma = {relu(self._sigma_x).item()}")
            print(f"... alpha = {self._alpha_x.item()}")
            print(f"... phi   = {self._phi_x.item()}")
            print(f"... beta1 = {relu(self._beta1_x).item()}")
            print(f"... beta2 = {relu(self._beta2_x).item()}")
            print("U parameters")
            print(f"... theta = {relu(self._theta_u).item()}")
            print(f"... mu    = {self._mu_u.item()}")
            print(f"... sigma = {relu(self._sigma_u).item()}")
            raise ValueError


        return f, g


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    """"
    Wrapper for the generator SDE. This defines the methods familiar to pytorch, such as forward.
    It wraps a GeneratorFunc instance and provides the mechanisms for numerically solving the SDE.
    """
    def __init__(self,
                 initial_condition: InitialCondition,
                 state_size: int,
                 mlp_size: int,
                 num_layers: int,
                 time_steps: int = None,
                 varnames: list[str] = None) -> None:
        """
        Constructor for the SDE wrapper.

        Parameters
        ----------
        initial_condition : InitialCondition
            Some initial condition to sample from.
        state_size : int
            The dimensionality of the state of the SDE.
        mlp_size : int
            The size of the hidden layers in the MLPs.
        num_layers : int
            The number of hidden layers in the MLPs.
        varnames : list[str]
            The names of the variables in the data, used to applying exogenous forcing functions to
            the drift and diffusion functions of the SDE by variable.
        add_forcing : dict
            A dictionary of forcing functions (implemented as torch.nn.Modules) indexed by variable name.
            These will be applied additively to the drift and diffusion functions of the SDE.
        mult_forcing : dict
            A dictionary of forcing functions (implemented as torch.nn.Modules) indexed by variable name.
            These will be applied multiplicatively to the drift and diffusion functions of the SDE.
        dawn_dusk_sampler : DawnDuskSampler
            A sampler for the dawn and dusk times of the solar data. These are appended to the system
            state and used to apply the solar forcing functions. This is a bit of a hack, but it's the
            best way I've found to
        """
        super().__init__()
        self._initial_condition = initial_condition
        self._state_size = state_size

        # The SDE itself.
        # self._func = GeneratorFunc(state_size, mlp_size, num_layers, varnames, add_forcing, mult_forcing)
        self._func = GeneratorFunc(state_size, mlp_size, num_layers, varnames)

        # default number of time steps to evaluate the SDE at
        # handy to not have to pass this in every time so we can use the same training infrastructure
        self._time_steps = time_steps

    def forward(self,
                initial_condition: torch.Tensor,
                ts: torch.Tensor = None,
                t_offset=None,
                return_fg=None) -> torch.Tensor:
        """
        Forward pass of the SDE.

        Parameters
        ----------
        initial_condition : torch.Tensor
            The initial SDE state.
        ts : torch.Tensor (optional)
            The times at which to evaluate the SDE. Has shape (t_size,).

        Returns
        -------
        ty_generated : torch.Tensor
            The values of the SDE at the given times. Has shape (batch_size, t_size, data_size).
        """
        device = initial_condition.device

        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and self._time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            ts = torch.arange(self._time_steps, device=device)

        batch_size = initial_condition.size(0)

        # Add a randomly sampled value to the initial condition to initialize the auxiliary SDE.
        u_mean = self._func._mu_u
        u_std  = self._func._sigma_u / (2 * self._func._theta_u)
        initial_condition = torch.cat([initial_condition, u_mean + u_std * torch.randn(batch_size, 1, device=device)], dim=1)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, initial_condition, ts, method='reversible_heun',
                                     dt=0.01, adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)
        xs = xs[..., :-1]

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        tx = torch.cat([ts, xs], dim=2)

        return tx

    def sample_latent(self,
                      num_samples: int) -> torch.Tensor:
        """
        Sample the initial noise distribution used to initialize the SDE.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.

        Returns
        -------
        torch.Tensor
            The samples. Has shape (num_samples, initial_noise_size).
        """
        return self._initial_condition.sample(num_samples)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        latent = self.sample_latent(num_samples)
        return self(latent)


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
###################
class DropFirstValue(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:].squeeze()


class DiscriminatorSimple(torch.nn.Module):
    def __init__(self, data_size: int, time_size: int, num_layers: int, num_units: int):
        super().__init__()
        layers = [DropFirstValue(), torch.nn.Flatten()]
        last_size = data_size * time_size
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(last_size, num_units))
            layers.append(torch.nn.ReLU())
            last_size = num_units
        layers.append(torch.nn.Linear(last_size, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
