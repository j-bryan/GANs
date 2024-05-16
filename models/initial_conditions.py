import torch


class InitialCondition:
    def __init__(self, output_size: int, device: torch.device | None = None):
        self.output_size = output_size
        self.device = device

    def sample(self, N: int) -> torch.Tensor:
        """
        Samples N initial conditions from the distribution.
        """
        raise NotImplementedError


class ConstantInitialCondition(InitialCondition):
    def __init__(self, value: float, output_size: int = 1, **kwargs):
        super().__init__(output_size, **kwargs)
        self.value = value

    def sample(self, N: int) -> torch.Tensor:
        return torch.full((N, self.output_size), self.value).to(self.device)


class DataInitialCondition(InitialCondition):
    """
    Samples from the initial condition of historical data to use as the initial condition for
    the generator.
    """
    def __init__(self, data, **kwargs):
        self.data = data
        if data.ndim == 1:
            self.data = self.data.unsqueeze(-1)
        output_size = data.shape[-1]
        super().__init__(output_size, **kwargs)

    def sample(self, N: int) -> torch.Tensor:
        """
        Randomly select N initial conditions from the data.
        """
        return self.data[torch.randint(0, len(self.data), (N,))].to(self.device)


class RandNormInitialCondition(InitialCondition):
    """
    Samples from a random normal distribution. Used when the model initialization is determined by
    sampling from a latent space, such as in a GAN.
    """
    def __init__(self, dim: int | tuple[int], **kwargs):
        """
        :param d: The dimension of the noise.
        """
        super().__init__(**kwargs)
        self.dim = dim

    def sample(self, N: int) -> torch.Tensor:
        return torch.randn(N, self.dim).to(self.device)
