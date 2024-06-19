import torch
import torchsde
import torchcde

from models.layers import FFNN, FFNNConfig
from models.preprocessing import Preprocessor

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
    return_ty: bool = False

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FFNNConfig):
                d.update(v.to_dict(prefix=k.split("_")[0]))
            else:
                d[k] = v
        return d


@dataclass
class CdeDiscriminatorConfig:
    data_size: int
    hidden_size: int
    mlp_size: int
    num_layers: int

    def to_dict(self, prefix=""):
        d = {}
        for k, v in self.__dict__.items():
            d[f"{prefix}_{k}"] = v
        return d


class GeneratorFunc(torch.nn.Module):
    """
        The generator SDE. The drift and diffusion functions as MLPs.
    """
    def __init__(self,
                 noise_size: int,
                 hidden_size: int,
                 drift_config: FFNNConfig,
                 diffusion_config: FFNNConfig) -> None:
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
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        # General drift and diffusion functions modeled with MLPs.
        self._drift = FFNN(**drift_config.to_dict())
        self._diffusion = FFNN(**diffusion_config.to_dict())

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

        f = self._drift(tx)
        g = self._diffusion(tx)
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
                                   config.diffusion_config)
        self._func.sde_type = config.sde_type
        self._func.noise_type = config.noise_type
        # MLP to map the state of the SDE to the space of the data.
        # self._readout = torch.nn.Linear(hidden_size, data_size)
        self._readout = FFNN(**config.readout_config.to_dict())
        # self._readout = torch.nn.Linear(config.readout_config.in_size, config.readout_config.out_size)

        # Whether or not to include time in the returned samples
        self.return_ty = config.return_ty

    def forward(self,
                init_noise: torch.Tensor,
                ts: torch.Tensor = None,
                gradients: bool = False,
                time_steps: int | None = None) -> torch.Tensor:
        """
        Forward pass of the SDE.

        Parameters
        ----------
        init_noise : torch.Tensor
            The initial noise used to initialize the SDE.
        ts : torch.Tensor (optional)
            The times at which to evaluate the SDE. Has shape (t_size,).

        Returns
        -------
        ty_generated : torch.Tensor
            The values of the SDE at the given times. Has shape (batch_size, t_size, data_size).
        """
        time_steps = time_steps or self._time_steps
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            ts = torch.arange(time_steps, device=init_noise.device)

        ###################
        # Actually solve the SDE.
        ###################
        batch_size = init_noise.size(0)
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        dt = ts[1] - ts[0]
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=dt,
                                     adjoint_method='adjoint_reversible_heun')
        # xs is returned as (time, batch_size, hidden_size)
        # Transpose to get (batch_size, time, hidden_size)
        xs = xs.transpose(0, 1)
        # concatenate time to last dimension of xs
        # Let's try adding a nighttime hours feature. We'll define nighttime hours as 8pm to 6am.
        # nighttime = ((ts.squeeze() % 24) > 20) | ((ts.squeeze() % 24) < 6)
        # nighttime = nighttime.unsqueeze(-1).float()
        # xs = torch.cat([ts, nighttime, xs], dim=2)
        ts = ts % 24  # normalize time to 24 hours for readout and return
        t_state = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        tx = torch.cat([t_state, xs], dim=2)
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

        if not self.return_ty:
            return ys

        # Append time to first dimension of ys
        ty = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1), ys], dim=2)

        return ty

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
        return torch.randn(num_samples, self._initial_noise_size)

    def sample(self, num_samples: int = 1, gradients: bool = False, time_steps: int | None = None) -> torch.Tensor:
        latent = self.sample_latent(num_samples).to(self._readout.parameters().__next__().data.device)
        return self(latent, gradients=gradients, time_steps=time_steps)


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
###################
class DiscriminatorFunc(torch.nn.Module):
    def __init__(self,
                 data_size: int,
                 hidden_size: int,
                 mlp_size: int,
                 num_layers: int) -> None:
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
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = FFNN(
            in_size=1 + hidden_size,
            out_size=hidden_size * (1 + data_size),
            num_units=mlp_size,
            num_hidden_layers=num_layers,
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
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, config: CdeDiscriminatorConfig) -> None:
        """
        Constructor for the discriminator wrapper.
        """
        super().__init__()

        # MLP to map from data space to latent space of discriminator CDE.
        self._initial = FFNN(
            1 + config.data_size,
            config.hidden_size,
            config.mlp_size,
            config.num_layers,
            activation='lipswish',
            final_activation='sigmoid'
        )
        # The discriminator CDE
        self._func = DiscriminatorFunc(config.data_size, config.hidden_size, config.mlp_size, config.num_layers)
        # Linear readout layer
        self._readout = torch.nn.Linear(config.hidden_size, 1)

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
