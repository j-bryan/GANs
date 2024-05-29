import torch
import torchsde

from models.layers import MLP


class InitialCondition:
    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError


class FixedInitialCondition(InitialCondition):
    def __init__(self, value: torch.Tensor | float):
        self.value = value

    def sample(self, n: int) -> torch.Tensor:
        if isinstance(self.value, torch.Tensor):
            return self.value.expand(n, -1)
        return self.value * torch.ones(n, 1)


class RandomInitialCondition(InitialCondition):
    def __init__(self, dim: int):
        self.dim = dim

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim)


class SampledInitialCondition(InitialCondition):
    def __init__(self, data: torch.Tensor):
        # data has shape (*, segment_size, data_size)
        # we need to keep only the first time point
        self.data = data[:, 0]

    def sample(self, n: int) -> torch.Tensor:
        # return a randomly sampled initial condition from the data, returning a tensor of shape
        # (n, data_size)
        idx = torch.randint(0, self.data.size(0), (n,))
        return self.data[idx]


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
                 varnames: list[str] = None,
                 add_forcing: dict[str, torch.nn.Module] = {},
                 mult_forcing: dict[str, torch.nn.Module] = {}) -> None:
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
        self.add_forcing = add_forcing
        self.mult_forcing = mult_forcing

        self._drift = MLP(
            in_size=1 + state_size,  # +1 for time dimension
            out_size=state_size,
            mlp_size=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='identity'
        )
        self._diffusion = MLP(
            in_size=1 + state_size,
            out_size=state_size,
            mlp_size=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='identity'
        )

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
        if torch.abs(x).max() > 1e6:
            # print(f"Large value in x!")
            # print(f"t: {t}")
            # print(f"x: {x}")
            # raise ValueError
            return torch.zeros_like(x), torch.zeros_like(x)

        # Last two dimensions of x are the dawn and dusk times
        t_dawn = x[..., -2]
        t_dusk = x[..., -1]
        kwargs = {'t_dawn': t_dawn, 't_dusk': t_dusk}
        # Remove the dawn and dusk times from x
        x = x[..., :-2]

        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        f = self._drift(tx)
        g = self._diffusion(tx)

        # We want the drift term to be positive for SOLAR, and we let the additional forcing function
        # determine the sign of the drift term from there.
        f[..., 2] = torch.nn.functional.sigmoid(f[..., 2])

        f_original = f.clone()
        g_original = g.clone()

        if t.ndim > 1:
            t = t.squeeze()

        for varname, forcing in self.mult_forcing.items():
            idx = self.varnames.index(varname)
            drift_mult, diffusion_mult = forcing(t, x[..., idx], **kwargs)
            f[:, idx] *= drift_mult
            g[:, idx] *= diffusion_mult

        for varname, forcing in self.add_forcing.items():
            idx = self.varnames.index(varname)
            drift_mult, diffusion_mult = forcing(t, x[..., idx], **kwargs)
            f[:, idx] += drift_mult
            g[:, idx] += diffusion_mult

        if torch.isnan(f).any() or torch.isnan(g).any():
            # Find which row has the NaN
            for i in range(f.size(0)):
                if torch.isnan(f[i]).any() or torch.isnan(g[i]).any():
                    print(f"t: {t[i]:2.1f}")
                    print(f"x: {[x[i, j].item() for j in range(x.size(1))]}")
                    print(f"f: {[f[i, j].item() for j in range(f.size(1))]}")
                    print(f"g: {[g[i, j].item() for j in range(g.size(1))]}")
                    print(f"f_original: {[f_original[i, j].item() for j in range(f_original.size(1))]}")
                    print(f"g_original: {[g_original[i, j].item() for j in range(g_original.size(1))]}")
                    break
            raise ValueError("NaN in f or g")

        # Append the dawn and dusk times back to the state. This can be done by concatenating a 0
        # to the end of f and g, causing the dawn and dusk times to be passed through the SDE unchanged.
        f = torch.cat([f, torch.zeros(f.size(0), 2, device=f.device)], dim=1)
        g = torch.cat([g, torch.zeros(g.size(0), 2, device=g.device)], dim=1)


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
                 varnames: list[str] = None,
                 add_forcing: dict[str, torch.nn.Module] = {},
                 mult_forcing: dict[str, torch.nn.Module] = {},
                 dawn_dusk_sampler=None) -> None:
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
        self._dawn_dusk_sampler = dawn_dusk_sampler

        # The SDE itself.
        self._func = GeneratorFunc(state_size, mlp_size, num_layers, varnames, add_forcing, mult_forcing)

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
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and self._time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            ts = torch.arange(self._time_steps, device=initial_condition.device)

        batch_size = initial_condition.size(0)
        if self._dawn_dusk_sampler is not None:
            t_dawn, t_dusk = self._dawn_dusk_sampler.sample(batch_size)
            # Pack the dawn and dusk times into the initial condition
            initial_condition = torch.cat([initial_condition, t_dawn.unsqueeze(1), t_dusk.unsqueeze(1)], dim=1)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, initial_condition, ts, method='reversible_heun',
                                     dt=0.1, adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)

        # Drop the dawn and dusk times from the state
        xs = xs[..., :-2]

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