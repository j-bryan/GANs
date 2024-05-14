import torch
from torch.autograd import Variable
from .trainer import Trainer

from utils.plotting import TrainingPlotter


class WGANLPTrainer(Trainer):
    """
    A WGAN with Lipschitz penalty term for the discriminator.
    """
    def __init__(self,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 penalty_weight: float = 10.0,
                 critic_iterations: int = 5,
                 plotter: TrainingPlotter = None,
                 device: str | None = None) -> None:
        """
        Constructor

        Parameters
        ----------
        generator : torch.nn.Module
            The generator model.
        discriminator : torch.nn.Module
            The discriminator model.
        g_optimizer : torch.optim.Optimizer
            The optimizer for the generator.
        d_optimizer : torch.optim.Optimizer
            The optimizer for the discriminator.
        penalty_weight : float
            The weight of the Lipschitz penalty term.
        critic_iterations : int
            The number of iterations to train the critic for each generator iteration.
        plotter : TrainingPlotter, optional
            The plotter to use for plotting training progress.
        device : str, optional
            The device to use for training. If None, a GPU is used if available, otherwise
            defaulting to CPU.
        """
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         critic_iterations, plotter, device)
        self.penalty_weight = penalty_weight

    def _critic_train_iteration(self, data: torch.Tensor) -> None:
        """
        Train the critic for one iteration.

        Parameters
        ----------
        data : torch.Tensor
            The real data to train on.
        """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data).to(self.device)
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        # d_loss --> 0 if same accuracy for real and generated data and gradient has norm 1
        # FIXME: For now we'll assume that the data and generated_data both have a time index in the
        # first entry of the last dimension. We need to drop this time index to calculate the denominator
        # in the penalty term.
        penalty = (d_real - d_generated).norm(1) / (data[:, :, 1:] - generated_data[:, :, 1:]).norm(2) - 1
        penalty[penalty < 0] = 0
        d_loss = d_generated.mean() - d_real.mean() + self.penalty_weight * penalty ** 2
        # Record loss
        d_loss.backward()
        self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()
