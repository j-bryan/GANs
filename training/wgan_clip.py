import torch
from torch.autograd import Variable
from .trainer import Trainer

from utils.plotting import TrainingPlotter


class WGANClipTrainer(Trainer):
    def __init__(self,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 weight_clip: float = 0.01,
                 critic_iterations: int = 5,
                 generator_iterations: int = 1,
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
        weight_clip : float
            The clipping value for the weights.
        critic_iterations : int
            The number of iterations to train the critic for each generator iteration.
        plotter : TrainingPlotter, optional
            The plotter to use for plotting training progress.
        device : str, optional
            The device to use for training. If None, a GPU is used if available, otherwise
            defaulting to CPU.
        """
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         critic_iterations, generator_iterations, plotter, device)
        self.clip = weight_clip

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
        d_loss = d_generated.mean() - d_real.mean()
        # Record loss
        d_loss.backward()
        self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()

        # Clip weights
        for p in self.D.parameters():
            p.data.clamp_(-self.clip, self.clip)
