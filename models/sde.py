import torch
import torchsde
import torchcde

from models.layers import MLP
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


class GeneratorFunc(torch.nn.Module):
    """
        The generator SDE. The drift and diffusion functions as MLPs.
    """
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self,
                 drift_func: torch.nn.Module,
                 diffusion_func: torch.nn.Module) -> None:
        """
        Constructor for the SDE

        Parameters
        ----------
        drift_func : torch.nn.Module
            The drift function of the SDE.
        diffusion_func : torch.nn.Module
            The diffusion function of the SDE.
        """
        super().__init__()
        self._drift = drift_func
        self._diffusion = diffusion_func

        # # General drift and diffusion functions modeled with MLPs.
        # self._drift = MLP(
        #     in_size=1 + hidden_size,  # +1 for time dimension
        #     out_size=hidden_size,
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='tanh'
        # )
        # self._diffusion = MLP(
        #     in_size=1 + hidden_size,
        #     out_size=hidden_size * noise_size,
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='tanh'
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
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        f = self._drift(tx)
        g = self._diffusion(tx)
        # reshape to match needed matrix dimensions
        g = g.view(x.size(0), self._diffusion.state_size, self._diffusion.noise_size)

        return f, g


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module, Preprocessor):
    """"
    Wrapper for the generator SDE. This defines the methods familiar to pytorch, such as forward.
    It wraps a GeneratorFunc instance and provides the mechanisms for numerically solving the SDE.
    """
    def __init__(self,
                 drift_func: torch.nn.Module,
                 diffusion_func: torch.nn.Module,
                 initial_condition: InitialCondition,
                 readout: torch.nn.Module | None = None,
                 initial_condition_embedding: torch.nn.Module | None = None,
                 time_steps: int = None) -> None:
        """
        Constructor for the SDE wrapper.

        Parameters
        ----------
        data_size : int
            The dimensionality of the data (number of variables).
        initial_condition : InitialCondition
            An object that samples a defined initial condition for the SDE.
        noise_size : int
            The dimensionality of the noise term in the SDE (the Brownian motion term).
        hidden_size : int
            The dimensionality of the state of the SDE.
        mlp_size : int
            The size of the hidden layers in the MLPs.
        num_layers : int
            The number of hidden layers in the MLPs.
        """
        super().__init__()
        Preprocessor.__init__(self)

        # An initial condition object that samples from some distribution.
        self._initial_condition = initial_condition if initial_condition_embedding is None else torch.nn.Identity()

        # MLP to map initial noise to the initial state of the SDE.
        self._initial = torch.nn.Identity() if initial_condition_embedding is None else initial_condition_embedding
        # self._initial = MLP(
        #     in_size=self._initial_condition.output_size,
        #     out_size=hidden_size,
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='sigmoid'
        # )
        # The SDE itself.
        self._func = GeneratorFunc(drift_func, diffusion_func)
        # MLP to map the state of the SDE to the space of the data.
        self._readout = torch.nn.Identity() if readout is None else readout

        # default number of time steps to evaluate the SDE at
        # handy to not have to pass this in every time so we can use the same training infrastructure
        self._time_steps = time_steps

    def forward(self,
                init_noise: torch.Tensor,
                ts: torch.Tensor = None) -> torch.Tensor:
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
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        if ts is None and self._time_steps is None:
            raise ValueError('Either pass in ts or set time_steps in the constructor!')
        elif ts is None:
            ts = torch.arange(self._time_steps, device=init_noise.device)

        ###################
        # Actually solve the SDE.
        ###################
        batch_size = init_noise.size(0)  # (batch_size, initial_noise_size)
        x0 = self._initial(init_noise)  # output has shape (batch_size, hidden_size)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        # interpolate to ensure we have data at the correct times
        ty_generated = torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))
        return ty_generated

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

    def transformed_sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.forward(x).transpose(1, 2).cpu().data.numpy()
        return self.preprocessor.inverse_transform(samples)


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
        self._module = func
        self._hidden_size = func.gen_state_size
        self._data_size = func.data_state_size + func.time_features
        # self._module = MLP(
        #     in_size=1 + hidden_size,
        #     out_size=hidden_size * (1 + data_size),
        #     mlp_size=mlp_size,
        #     num_layers=num_layers,
        #     activation='lipswish',
        #     final_activation='tanh'
        # )

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
        self._initial = torch.nn.Identity() if inital_condition_embedding is None else inital_condition_embedding
        # self._initial = MLP(
        #     1 + data_size,
        #     hidden_size,
        #     mlp_size,
        #     num_layers,
        #     activation='lipswish',
        #     final_activation='sigmoid'
        # )
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
        return score
