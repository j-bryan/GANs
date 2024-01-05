import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from .trainer import Trainer


class SDEGANTrainer(Trainer):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 weight_clip=0.01, critic_iterations=5, use_cuda=False,
                 early_stopping=None, plotter=None):
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         critic_iterations, use_cuda, early_stopping, plotter)
        self.clip = weight_clip

    def _critic_train_iteration(self, data):
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        # d_loss --> 0 if same accuracy for real and generated data
        d_loss = d_generated.mean() - d_real.mean()
        # Record loss
        d_loss.backward()
        self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()

        # Clip weights
        for p in self.D.parameters():
            p.data.clamp_(-self.clip, self.clip)
