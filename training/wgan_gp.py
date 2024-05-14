import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from .trainer import Trainer
from utils.plotting import TrainingPlotter


class WGANGPTrainer(Trainer):
    """ Trainer for Wasserstein GAN with gradient penalty """
    def __init__(self,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 penalty_weight: float = 10,
                 critic_iterations: int = 5,
                 plotter: TrainingPlotter | None = None,
                 device: str | None = None) -> None:
        """
        Constructor.

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
        penalty_weight : int
            The weight for the gradient penalty.
        critic_iterations : int
            The number of iterations to train the critic for each generator iteration.
        plotter : TrainingPlotter, optional
            The plotter to use for plotting training progress.
        device : str, optional
            The device to use for training. If None, a GPU is used if available, otherwise
            defaulting to CPU.
        """
        super().__init__(generator, discriminator, g_optimizer, d_optimizer, critic_iterations, plotter)
        self.penalty_weight = penalty_weight
        self.losses |= {'GP': [], 'gradient_norm': []}  # add gradient penalty terms to losses dict

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

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        # d_loss --> 0 if same accuracy for real and generated data and gradient has norm 1
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        # Record loss
        d_loss.backward()
        self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()

    def _gradient_penalty(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient penalty for the discriminator.

        Parameters
        ----------
        real_data : torch.Tensor
            The real data.
        generated_data : torch.Tensor
            The generated data.

        Returns
        -------
        penalty : torch.Tensor
            The gradient penalty.
        """
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        grad_output = torch.ones(prob_interpolated.size()).to(self.device)
        gradients = torch_grad(outputs=prob_interpolated,
                               inputs=interpolated,
                               grad_outputs=grad_output,
                               create_graph=True,
                               retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)
        self.losses['gradient_norm'].append(gradients_norm.mean().data)

        # Return gradient penalty
        penalty = self.penalty_weight * ((gradients_norm - 1) ** 2).mean()
        return penalty
