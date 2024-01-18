import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from .trainer import Trainer


class WGANGPTrainer(Trainer):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gp_weight=10, critic_iterations=5, early_stopping=None, plotter=None):
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         critic_iterations, early_stopping, plotter)
        self.gp_weight = gp_weight
        self.losses |= {'GP': [], 'gradient_norm': []}

    def _critic_train_iteration(self, data):
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

    def _gradient_penalty(self, real_data, generated_data):
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
        penalty = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return penalty
