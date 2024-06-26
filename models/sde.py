import torch
import torchsde
import torchcde

from models.layers import FFNN, FFNNConfig
from models.preprocessing import Preprocessor
from models.initial_conditions import InitialCondition


class DriftMLP(torch.nn.Module):
    def __init__(self,
                 state_size: int,
                 mlp_size: int,
                 num_layers: int,
                 time_features: int = 1,
                 activation='lipswish',
                 final_activation='tanh'):
        super().__init__()
        self.state_size = state_size
        self.time_features = time_features
        self.model = MLP(in_size=state_size + time_features,
                         out_size=state_size,
                         mlp_size=mlp_size,
                         num_layers=num_layers,
                         activation=activation,
                         final_activation=final_activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class DiffusionMLP(torch.nn.Module):
    def __init__(self,
                 state_size: int,
                 noise_size: int,
                 mlp_size: int,
                 num_layers: int,
                 time_features: int = 1,
                 activation='lipswish',
                 final_activation='tanh'):
        super().__init__()
        self.state_size = state_size
        self.noise_size = noise_size
        self.model = MLP(in_size=state_size + time_features,
                         out_size=state_size * noise_size,
                         mlp_size=mlp_size,
                         num_layers=num_layers,
                         activation=activation,
                         final_activation=final_activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

from dataclasses import dataclass


@dataclass
class SdeGeneratorConfig:
    # SDE parameters
    noise_size: int
    hidden_size: int
    init_noise_size: int

    # Observed data parameters
    data_size: int

    # Component model configurations
    drift_config: FFNNConfig
    diffusion_config: FFNNConfig
    embed_config: FFNNConfig
    readout_config: FFNNConfig

    # Default SDE parameters
    sde_type: str = "stratonovich"
    noise_type: str = "diagonal"
    time_size: int = 1
    time_steps: int = 24
    time_dependent_readout: bool = False
    time_dependent_drift: bool = False
    time_dependent_diffusion: bool = False

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FFNNConfig):
                d.update(v.to_dict(prefix=k.split("_")[0]))
            else:
                d[k] = v
        return d


class GeneratorFunc(torch.nn.Module):
    """
        The generator SDE. The drift and diffusion functions as MLPs.
    """
    def __init__(self,
                 noise_size: int,
                 hidden_size: int,
                 drift_config: FFNNConfig,
                 diffusion_config: FFNNConfig,
                 time_dependent_drift: bool,
                 time_dependent_diffusion: bool) -> None:
        """
        Constructor for the SDE

        Parameters
        ----------
        noise_size : int
            The dimensionality of the noise term.
        hidden_size : int
            The dimensionality of the hidden state.
        """
        super().__init__()
        self._drift = drift_func
        self._diffusion = diffusion_func

        # General drift and diffusion functions modeled with MLPs.
        self._drift = FFNN(**drift_config.to_dict())
        self._diffusion = FFNN(**diffusion_config.to_dict())

        self.time_dependent_drift = time_dependent_drift
        self.time_dependent_diffusion = time_dependent_diffusion

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
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        f = self._drift(tx) if self.time_dependent_drift else self._drift(x)
        g = self._diffusion(tx) if self.time_dependent_diffusion else self._diffusion(x)
        # f = self._drift(tx)
        # g = self._diffusion(tx)
        # reshape to match needed matrix dimensions
        if self.noise_type == "general":
            g = g.view(x.size(0), self._hidden_size, self._noise_size)

        return f, g


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    """"
    Wrapper for the generator SDE. This defines the methods familiar to pytorch, such as forward.
    It wraps a GeneratorFunc instance and provides the mechanisms for numerically solving the SDE.
    """
    def __init__(self, config) -> None:
        """
        Constructor for the SDE wrapper.
        """
        super().__init__()
        self._initial_noise_size = config.init_noise_size
        self._hidden_size = config.hidden_size
        # default number of time steps to evaluate the SDE at
        # handy to not have to pass this in every time so we can use the same training infrastructure
        self._time_steps = config.time_steps

        # MLP to map initial noise to the initial state of the SDE.
        self._initial = FFNN(**config.embed_config.to_dict())
        # The SDE itself.
        self._func = GeneratorFunc(config.noise_size,
                                   config.hidden_size,
                                   config.drift_config,
                                   config.diffusion_config,
                                   config.time_dependent_drift,
                                   config.time_dependent_diffusion)
        self._func.sde_type = config.sde_type
        self._func.noise_type = config.noise_type
        # MLP to map the state of the SDE to the space of the data.
        # self._readout = torch.nn.Linear(hidden_size, data_size)
        self._readout = FFNN(**config.readout_config.to_dict())
        # self._readout = torch.nn.Linear(config.readout_config.in_size, config.readout_config.out_size)
        self._time_dependent_readout = config.time_dependent_readout

    def forward(self,
                init_noise: torch.Tensor,
                ts: torch.Tensor = None,
                gradients: bool = False,
                hidden: bool = False) -> torch.Tensor:
        """
        Forward pass of the SDE.

        Parameters
        ----------
        init_noise : torch.Tensor
            The initial noise used to initialize the SDE.
        ts : torch.Tensor (optional)
            The times at which to evaluate the SDE. Has shape (t_size,).
        gradients : bool
            Whether to calculate gradients in the observed space of Y_t.
        hidden : bool
            Whether to return the hidden states of the SDE.

        Returns
        -------
        ty_generated : torch.Tensor
            The values of the SDE at the given times. Has shape (batch_size, t_size, data_size).
        """
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and self._time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            # Randomly vary the number of days to evaluate the SDE at
            # ts = torch.arange(torch.randint(low=1, high=3, size=(1,)).item() * self._time_steps, device=init_noise.device)
            ts = torch.arange(self._time_steps, device=init_noise.device)

        ###################
        # Actually solve the SDE.
        ###################
        batch_size = init_noise.size(0)  # (batch_size, initial_noise_size)
        x0 = self._initial(init_noise)  # output has shape (batch_size, hidden_size)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        dt = ts[1] - ts[0]
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=dt,
                                     adjoint_method='adjoint_reversible_heun')
        # xs is returned as (time, batch_size, hidden_size)
        # Transpose to get (batch_size, time, hidden_size)
        xs = xs.transpose(0, 1)
        # Keep only the last self._time_steps time steps of xs
        # xs = xs[:, -self._time_steps:, :]
        # concatenate time to last dimension of xs
        if self._time_dependent_readout:
            t_state = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
            tx = torch.cat([t_state, xs], dim=2)
        else:
            tx = xs
        # Apply readout to batches of hidden states one time step at a time
        ys = self._readout(tx)

        #########################################
        if gradients:
            # Use the generated samples to calculate gradients in the observed space of Y_t
            f = torch.zeros(batch_size, self._time_steps, self._hidden_size)
            if self._func.noise_type == "diagonal":
                g = torch.zeros(batch_size, self._time_steps, self._hidden_size)
                for i in range(self._time_steps):
                    f[:, i], g[:, i] = self._func.f_and_g(ts[i], xs[:, i])
                # Since we're treating time like a state variable here, we need to add a 1 to the beginning
                # of the f tensor and a 0 to the beginning of the g tensor.
                f = torch.cat([torch.ones(batch_size, self._time_steps, 1), f], dim=2).unsqueeze(-1)
                g = torch.cat([torch.zeros(batch_size, self._time_steps, 1), g], dim=2).unsqueeze(-1)
            elif self._func.noise_type == "general":
                g = torch.zeros(batch_size, self._time_steps, self._hidden_size, self._func._noise_size)
                for i in range(self._time_steps):
                    f[:, i], g[:, i] = self._func.f_and_g(ts[i], xs[:, i])
                # Since we're treating time like a state variable here, we need to add a 1 to the beginning
                # of the f tensor and a 0 to the beginning of the g tensor.
                f = torch.cat([torch.ones(batch_size, self._time_steps, 1), f], dim=2).unsqueeze(-1)
                g = torch.cat([torch.zeros(batch_size, self._time_steps, 1, g.size(-1)), g], dim=2)

            readout_jacobian = torch.zeros((batch_size, self._time_steps, ys.size(-1), self._hidden_size + 1))
            # We need to calculate the jacobian of the readout layer with respect to the hidden state
            # of the SDE for each time step and each batch.
            for i in range(batch_size):
                for j in range(self._time_steps):
                    readout_jacobian[i, j] = torch.autograd.functional.jacobian(self._readout, tx[i, j, :]).squeeze()
            # With readout_jacobian of shape (batch_size, time_steps, data_size, hidden_size + 1)
            drift = torch.matmul(readout_jacobian, f).squeeze(-1)
            diffusion = torch.matmul(readout_jacobian, g).squeeze(-1)
        #########################################

        if gradients:
            return ys, drift, diffusion

        if hidden:
            return ys, xs

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
        return self._initial_condition.sample(num_samples)

    def sample(self, num_samples: int = 1, gradients: bool = False) -> torch.Tensor:
        latent = self.sample_latent(num_samples).to(self._readout.parameters().__next__().data.device)
        return self(latent, gradients=gradients)


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
###################
class DiscriminatorMLP(torch.nn.Module):
    def __init__(self,
                 gen_state_size: int,
                 data_state_size: int,
                 mlp_size: int,
                 num_layers: int,
                 time_features: int = 1,
                 activation: str = 'lipswish',
                 final_activation: str = 'sigmoid') -> None:
        super().__init__()
        self.gen_state_size = gen_state_size
        self.data_state_size = data_state_size
        self.time_features = time_features
        self.model = MLP(in_size=time_features + gen_state_size,
                         out_size=gen_state_size * (time_features + data_state_size),
                         mlp_size=mlp_size,
                         num_layers=num_layers,
                         activation=activation,
                         final_activation=final_activation)
        print(self.model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, func) -> None:
        """
        Constructor for the discriminator CDE.

        Parameters
        ----------
        data_size : int
            The dimensionality of the data (number of variables).
        hidden_size : int
            The dimensionality of the SDE state vector. This does not need to match the dimensionality
            of the data.
        mlp_size : int
            The size of the hidden layers in the MLPs.
        num_layers : int
            The number of hidden layers in the MLPs.
        """
        super().__init__()

        # tanh is important for model performance
        self._module = FFNN(
            in_size=1 + hidden_size,
            out_size=hidden_size * (1 + data_size),
            num_units=mlp_size,
            num_layers=num_layers,
            activation='lipswish',
            final_activation='tanh'
        )

    def forward(self,
                t: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator CDE.

        Parameters
        ----------
        t : torch.Tensor
            The time.
        h : torch.Tensor
            The current state of the CDE.

        Returns
        -------
        torch.Tensor
            The next state of the CDE.
        """
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, self._data_size)


class Discriminator(torch.nn.Module):
    # def __init__(self,
    #              data_size: int,
    #              hidden_size: int,
    #              mlp_size: int,
    #              num_layers: int):
    def __init__(self,
                 discr_func: torch.nn.Module,
                 inital_condition_embedding: torch.nn.Module | None = None,
                 readout: torch.nn.Module | None = None) -> None:
        """
        Constructor for the discriminator wrapper.

        Parameters
        ----------
        data_size : int
            The dimensionality of the data (number of variables).
        hidden_size : int
            The dimensionality of the CDE state vector. This does not need to match the dimensionality
            of the data.
        mlp_size : int
            The size of the hidden layers in the MLPs.
        num_layers : int
            The number of hidden layers in the MLPs.
        """
        super().__init__()

        # MLP to map from data space to latent space of discriminator CDE.
        self._initial = FFNN(
            1 + data_size,
            hidden_size,
            mlp_size,
            num_layers,
            activation='lipswish',
            final_activation='sigmoid'
        )
        # The discriminator CDE
        # self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._func = DiscriminatorFunc(discr_func)
        # Linear readout layer
        # self._readout = torch.nn.Linear(hidden_size, 1)
        self._readout = torch.nn.Identity() if readout is None else readout

    def forward(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        ys_coeffs : torch.Tensor
            The data to evaluate the discriminator at. Has shape (batch_size, t_size, 1 + data_size).
            The +1 corresponds to time.

        Returns
        -------
        score : torch.Tensor
            The discriminator scores for the given data. Has shape (batch_size,).
        """
        # Treat time as just another channel in the CDE solution. This allows the CDe to more easily
        # handle cases of irregularly sampled data.
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun',
                             backend='torchsde', dt=1.0, adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()


class DropFirstValue(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:].squeeze()


class DiscriminatorSimple(torch.nn.Module):
    def __init__(self, config: FFNNConfig):
        super().__init__()
        # self.model = torch.nn.Sequential(DropFirstValue(), FFNN(**config.to_dict()))
        self.model = torch.nn.Sequential(torch.nn.Flatten(), FFNN(**config.to_dict()))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
