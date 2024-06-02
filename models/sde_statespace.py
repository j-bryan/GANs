import torch
import torchsde

from models.sde_no_embed import InitialCondition
from models.layers import FFNN


class GeneratorFunc(torch.nn.Module):
    """
        The generator SDE. The drift and diffusion functions as MLPs.
    """
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self,
                 state_size: int,
                 noise_size: int,
                 f_num_units: int,
                 f_num_layers: int,
                 g_num_units: int,
                 g_num_layers: int,
                 aux_size: int = 0,
                 varnames: list[str] = None,
                 aux_ou: bool = False) -> None:
        """
        Constructor for the SDE

        Parameters
        ----------
        state_size : int
            The dimensionality of the state (number of variables).
        noise_size : int
            The dimensionality of the Brownian motion. For general noise, this does not need to be the same as the
            state size.
        f_num_units : int
            The size of the hidden layers in the drift FFNN.
        f_num_layers : int
            The number of hidden layers in the drift FFNN.
        g_num_units : int
            The size of the hidden layers in the diffusion FFNN.
        g_num_layers : int
            The number of hidden layers in the diffusion FFNN.
        aux_size : int (default=0)
            The number of auxiliary variables to include in the state.
        varnames : list[str]
            The names of the variables in the data, used to applying exogenous forcing functions to
            aux_ou drift and diffusion functions of the SDE by variable.
        aux_ou : bool (default=False)
            Whether or not to model the auxiliary variables as Ornstein-Uhlenbeck processes.
            If True, fixed parameters mu=0 and sigma=1 are used, with theta being trainable. The scale and
            location of the Ornstein-Uhlenbeck processes are adjusted by a Linear layer when interacting
            with the main variables.
            If False, the drift and diffusion functions are modeled by the FFNNs of the generator.
            If choosing to model the auxiliary variables as Ornstein-Uhlenbeck processes, we require that
            the noise_size is equal to the state_size.
        """
        super().__init__()
        # General drift and diffusion functions modeled with MLPs.
        self.state_size = state_size
        self.noise_size = noise_size
        self.aux_size = aux_size
        self.varnames = varnames
        self.aux_ou = aux_ou

        if self.aux_ou:
            self._theta = torch.nn.Parameter(torch.ones(aux_size), requires_grad=True)
            if noise_size != state_size + aux_size:
                raise ValueError("To model auxiliary variables as Ornstein-Uhlenbeck processes, noise_size must be equal to state_size + aux_size. "
                                 f"Got noise_size = {noise_size}, state_size = {state_size}, and aux_size = {aux_size}.")
            self.noise_type = "diagonal"

        # If force_aux_ou is False, we need to expand the drift and diffusion FFNNs
        input_size = 1 + state_size + aux_size  # +1 for time dimension
        out_size   = state_size + aux_size * (aux_ou == False)

        self._drift = FFNN(
            in_size=input_size,
            out_size=out_size,
            num_units=f_num_units,
            num_layers=f_num_layers,
            activation='lipswish',
            final_activation='identity'
        )
        diff_out_size = state_size + aux_size if self.noise_type == "diagonal" else (state_size + aux_size) * noise_size
        self._diffusion = FFNN(
            in_size=input_size,
            out_size=diff_out_size,
            num_units=g_num_units,
            num_layers=g_num_layers,
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
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        if not self.aux_ou:
            # Just use the FFNNs to model the whole thing
            f = self._drift(tx)
            g = self._diffusion(tx)
            # reshape to match needed matrix dimensions
            g = g.view(x.size(0), self.state_size + self.aux_size, self.noise_size)
        else:
            # We're going to apply the FFNN to just the first state_size variables of x.
            # The rest will be modeled as Ornstein-Uhlenbeck processes.
            f = torch.zeros_like(x)
            g = torch.zeros_like(x)

            f[:, :self.state_size] = self._drift(tx)
            g[:, :self.state_size] = self._diffusion(tx)

            # Apply the Ornstein-Uhlenbeck process to the auxiliary variables
            f[:, self.state_size:] = -self._theta * x[:, self.state_size:]
            g[:, self.state_size:] = 1.0

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
                 time_size: int,
                 state_size: int,
                 noise_size: int,
                 f_num_units: int,
                 f_num_layers: int,
                 g_num_units: int,
                 g_num_layers: int,
                 aux_size: int = 0,
                 varnames: list[str] = None,
                 aux_ou: bool = False,
                 dawn_dusk_sampler=None) -> None:
        super().__init__()
        self._initial_condition = initial_condition
        self._state_size = state_size
        self._dd_sampler = dawn_dusk_sampler
        self._varnames = varnames

        # The SDE itself.
        # self._func = GeneratorFunc(state_size, mlp_size, num_layers, varnames, add_forcing, mult_forcing)
        self._func = GeneratorFunc(state_size, noise_size, f_num_units, f_num_layers, g_num_units, g_num_layers, aux_size, varnames, aux_ou)

        if not aux_ou:
            # We need to define a FFNN to embed the initial noise into the state space since we don't know what kind of
            # process it is; we need to learn it. We'll force the model to use the same number of random values as we
            # have auxiliary state variables. We'll also define a sensibly small FFNN to do this. TODO: make this a
            # user-defined parameter.
            self._noise_embedder = FFNN(
                in_size=aux_size,
                out_size=aux_size,
                num_layers=2,
                num_units=32,
                activation='lipswish',
                final_activation='tanh'
            )

        # default number of time steps to evaluate the SDE at
        # handy to not have to pass this in every time so we can use the same training infrastructure
        self._time_steps = time_size

    def _sample_OU_initial_condition(self, batch_size: int) -> torch.Tensor:
        """
        Sample initial conditions for the Ornstein-Uhlenbeck process auxiliary variables using the parameters.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate.

        Returns
        -------
        torch.Tensor
            The samples. Has shape (batch_size, state_size).
        """
        theta = self._func._theta
        return torch.randn(batch_size, self._func.aux_size) / (2 * theta)

    def _transform_outputs(self, xs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the output transformations to the data.

        Parameters
        ----------
        xs : torch.Tensor
            The output of the SDE. Has shape (batch_size, t_size, data_size).

        Returns
        -------
        torch.Tensor
            The transformed data. Has shape (batch_size, t_size, data_size).
        """
        # Before we return, we need to apply the output transformations to the data.
        #   - Load: none
        #   - Wind: sigmoid (0, 1)
        #   - Solar: sigmoid (0, 1) times sin(pi (t - t_m) / (t_n - t_m))
        ys = torch.zeros_like(xs)
        dawn = kwargs["dawn"]
        dusk = kwargs["dusk"]

        # Load: no transformation
        load_idx = self._varnames.index("TOTALLOAD")
        ys[..., load_idx] = xs[..., load_idx]

        # Wind: sigmoid (0, 1)
        wind_idx = self._varnames.index("WIND")
        ys[..., wind_idx] = torch.sigmoid(xs[..., wind_idx])

        # Solar: sigmoid (0, 1) times sin(pi (t - t_m) / (t_n - t_m))
        solar_idx = self._varnames.index("SOLAR")
        t = torch.arange(xs.size(1), device=xs.device).unsqueeze(0).expand(xs.size(0), xs.size(1))
        t_m = dawn.unsqueeze(-1)
        t_n = dusk.unsqueeze(-1)
        ys[..., solar_idx] = torch.sigmoid(xs[:, :, solar_idx]) * torch.sin(torch.pi * (t - t_m) / (t_n - t_m)) \
                             * (t >= t_m) * (t <= t_n)

        return ys

    def forward(self,
                initial_condition: torch.Tensor,
                ts: torch.Tensor = None) -> torch.Tensor:
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
        init_cond = self._initial_condition.get_inds(initial_condition)
        dawn, dusk = self._dd_sampler.get_inds(initial_condition)
        device = init_cond.device

        dawn = dawn.to(device)
        dusk = dusk.to(device)

        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and self._time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            ts = torch.arange(self._time_steps, device=device)

        batch_size = init_cond.size(0)

        if self._func.aux_ou:
            # Sample the auxiliary variables
            aux_init = self._sample_OU_initial_condition(batch_size)
            init_cond = torch.cat([init_cond, aux_init], dim=1)
        else:
            # Generate random noise, then use a FFNN to embed it into the state space
            aux_size = self._func.aux_size
            noise = torch.randn((batch_size, aux_size), device=device)
            aux_init = self._noise_embedder(noise)
            init_cond = torch.cat([init_cond, aux_init], dim=1)

        # We're using a sigmoid activation function in the output layer for WIND, so we need to apply
        # the inverse transformation to the initial condition.
        ic = torch.zeros_like(init_cond)
        ic[..., self._varnames.index("TOTALLOAD")] = init_cond[..., self._varnames.index("TOTALLOAD")]
        ic[..., self._varnames.index("WIND")] = torch.logit(init_cond[..., self._varnames.index("WIND")])
        ic[..., self._varnames.index("SOLAR")] = init_cond[..., self._varnames.index("SOLAR")]
        ic[..., self._func.state_size:] = init_cond[..., self._func.state_size:]

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, ic, ts, method='reversible_heun',
                                     dt=0.01, adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)

        # Readout only the state variables, not the auxiliary variables
        xs = xs[:, :, :self._state_size]

        # ###################
        # # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        # ###################
        # ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        # tx = torch.cat([ts, xs], dim=2)

        # return tx

        # Before we return, we need to apply the output transformations to the data.
        #   - Load: none
        #   - Wind: clamp (0, 1)
        #   - Solar: sigmoid (0, 1) times sin(pi (t - t_m) / (t_n - t_m))
        ys = self._transform_outputs(xs, dawn=dawn, dusk=dusk)

        return ys

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
        # return self._initial_condition.sample(num_samples)
        # return random indices to pull from the data
        return torch.randint(0, self._initial_condition.shape[0], (num_samples,))

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
        # layers = [DropFirstValue(), torch.nn.Flatten()]
        layers = [torch.nn.Flatten()]
        last_size = data_size * time_size
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(last_size, num_units))
            layers.append(torch.nn.ReLU())
            last_size = num_units
        layers.append(torch.nn.Linear(last_size, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
