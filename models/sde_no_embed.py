import torch
import torchsde

from models.layers import FFNN


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
    def __init__(self, dim: int, low: float = 0.0, high: float = 1.0):
        self.dim = dim
        self.low = low
        self.high = high

    def sample(self, n: int) -> torch.Tensor:
        return torch.rand(n, self.dim) * (self.high - self.low) + self.low


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

    def get_inds(self, idx):
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape


class ColumnwiseInitialCondition(InitialCondition):
    def __init__(self, conditions: tuple[InitialCondition, list[int]], device):
        """
        :param conditions: A tuple of (initial condition, column indices) pairs.
        """
        self.conditions = conditions
        self.num_columns = max([max(cols) for _, cols in conditions]) + 1
        self.device = device

    def sample(self, n: int) -> torch.Tensor:
        ic = torch.zeros(n, self.num_columns).to(self.device)
        for cond, cols in self.conditions:
            ic[:, cols] = cond.sample(n).to(self.device)
        return ic

    def get_inds(self, idx):
        ic = torch.zeros(idx.size(0), self.num_columns).to(self.device)
        for cond, cols in self.conditions:
            if hasattr(cond, "get_inds"):
                ic[:, cols] = cond.get_inds(idx).to(self.device)
            else:
                ic[:, cols] = cond.sample(idx.size(0)).to(self.device)
        return ic

    @property
    def shape(self):
        cond_shape = None
        for cond in self.conditions:
            if hasattr(cond, "shape"):
                if cond_shape is None:
                   cond_shape = cond.shape
                else:
                    assert cond_shape[:-1] == cond.shape[:-1]
        return cond_shape


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

        self._drift = FFNN(
            in_size=1 + state_size,  # +1 for time dimension
            out_size=state_size,
            num_units=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='identity'
        )
        self._diffusion = FFNN(
            in_size=1 + state_size,
            out_size=state_size,
            num_units=mlp_size,
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
            drift_add, diffusion_add = forcing(t, x[..., idx], **kwargs)
            f[:, idx] += drift_add
            g[:, idx] += diffusion_add

        # Append the dawn and dusk times back to the state. This can be done by concatenating a 0
        # to the end of f and g, causing the dawn and dusk times to be passed through the SDE unchanged.
        f = torch.cat([f, torch.zeros(f.size(0), 2, device=f.device)], dim=1)
        g = torch.cat([g, torch.zeros(g.size(0), 2, device=g.device)], dim=1)

        return f, g


class GeneratorFuncCustom(torch.nn.Module):
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

        # Solar mean-reversion parameters
        self._solar_theta = torch.nn.Parameter(torch.tensor(1.0))
        self._solar_mu = torch.nn.Parameter(torch.tensor(0.5))
        self._solar_sigma = torch.nn.Parameter(torch.tensor(1.0))

        # Wind mean-reversion parameters
        self._wind_theta = torch.nn.Parameter(torch.tensor(1.0))
        self._wind_mu = torch.nn.Parameter(torch.tensor(0.5))
        self._wind_sigma = torch.nn.Parameter(torch.tensor(1.0))

        self._drift = FFNN(
            in_size=1 + state_size,  # +1 for time dimension
            out_size=state_size,
            num_units=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='tanh'
        )
        self._diffusion = FFNN(
            in_size=1 + state_size,
            out_size=state_size,
            num_units=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='tanh'
        )

    def _solar_fg(self, t, x):
        x_solar = x[:, self.varnames.index("SOLAR")]
        drift = torch.square(self._solar_theta) * (self._solar_mu - x_solar)
        diffusion = torch.square(self._solar_sigma) * x_solar * (1 - x_solar)
        return drift, diffusion

    def _wind_fg(self, t, x):
        x_wind = x[:, self.varnames.index("WIND")]
        drift = torch.square(self._wind_theta) * (self._wind_mu - x_wind)
        diffusion = torch.square(self._wind_sigma) * x_wind * (1 - x_wind)
        return drift, diffusion

    def _load_fg(self, t, x):
        # We're going to leave this one to the FFNN entirely.
        return 1., 1.

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
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        f = torch.zeros_like(x)
        g = torch.zeros_like(x)

        # Fill in the drift and diffusion terms one by one
        load_idx = self.varnames.index("TOTALLOAD")
        wind_idx = self.varnames.index("WIND")
        solar_idx = self.varnames.index("SOLAR")

        f[:, load_idx], g[:, load_idx] = self._load_fg(t, x)
        f[:, wind_idx], g[:, wind_idx] = self._wind_fg(t, x)
        f[:, solar_idx], g[:, solar_idx] = self._solar_fg(t, x)

        # Add FFNN terms to f and g to allow for more complex dynamics that we don't model explicitly.
        f *= self._drift(tx)
        g *= self._diffusion(tx)

        if torch.any(f.abs() > 1e6) or torch.any(g.abs() > 1e6):
            # Find which row has the NaN
            for i in range(f.size(0)):
                if torch.any(f[i].abs() > 1e6) or torch.any(g[i].abs() > 1e6):
                    print(f"t: {t[i].item():2.1f}")
                    print(f"x: {[x[i, j].item() for j in range(x.size(1))]}")
                    print(f"f: {[f[i, j].item() for j in range(f.size(1))]}")
                    print(f"g: {[g[i, j].item() for j in range(g.size(1))]}")
                    raise ValueError("NaN in f or g")
            print("Couldn't find the NaN???")

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
        # self._func = GeneratorFunc(state_size, mlp_size, num_layers, varnames, add_forcing, mult_forcing)
        self._func = GeneratorFuncCustom(state_size, mlp_size, num_layers, varnames)

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
        # if self._dawn_dusk_sampler is not None:
        #     t_dawn, t_dusk = self._dawn_dusk_sampler.sample(batch_size)
        #     # Pack the dawn and dusk times into the initial condition
        #     initial_condition = torch.cat([initial_condition, t_dawn.unsqueeze(1).to(device), t_dusk.unsqueeze(1).to(device)], dim=1)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, initial_condition, ts, method='reversible_heun',
                                     dt=0.01, adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)

        # Drop the dawn and dusk times from the state
        # xs = xs[..., :-2]

        # Make any solar value before dawn or after dusk zero
        xs = torch.concat([xs, torch.zeros((batch_size, len(ts), 1), device=device)], dim=2)
        dawn, dusk = self._dawn_dusk_sampler.sample(batch_size)
        dawn = dawn.unsqueeze(1).to(device)
        dusk = dusk.unsqueeze(1).to(device)
        ts_tiled = ts.unsqueeze(0).expand(batch_size, ts.size(0))
        solar_idx = self._func.varnames.index("SOLAR")
        # Apply a quadratic transformation to the remaining data values
        h = (dawn + dusk) / 2
        k = 1.0
        a = -4 * k / (dusk - dawn) ** 2
        xs[..., solar_idx + 1] = xs[..., solar_idx] * (a * torch.square(ts_tiled - h) + k) * (ts_tiled >= dawn) * (ts_tiled <= dusk)
        xs = xs[..., [0, 1, 3]]

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
