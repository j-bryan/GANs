from tqdm import trange, tqdm
from torch.autograd import Variable


class Trainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 critic_iterations=5, use_cuda=False, early_stopping=None, plotter=None):
        self.G = generator
        self.D = discriminator
        self.G_opt = g_optimizer
        self.D_opt = d_optimizer
        self.losses = {'G': [], 'D': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.critic_iterations = critic_iterations

        self._fixed_latent = None
        self.print_every = 0  # by default, don't print or plot anything
        self.plot_every = 0
        self.plotter = plotter

        self.early_stopping = early_stopping
        self.check_stopping = False if early_stopping is None else True

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

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
        # d_loss --> 0 if same accuracy for real and generated data and gradient has norm 1
        d_loss = d_generated.mean() - d_real.mean()
        # Record loss
        d_loss.backward()
        self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()

    def _generator_train_iteration(self, data):
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        # The generator outputs a number between 0 and 1, corresponding to the probability that the sample is real.
        # If the generator is performing well, the discriminator is fooled by the generated data and produces a high value.
        # We want to maximize that, so we flip the sign.
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data.item())

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            if (i + 1) % self.critic_iterations == 0:  # only update generator every critic_iterations iterations
                self._generator_train_iteration(data)

    def train(self, data_loader, epochs, print_every=None, plot_every=None):
        self.print_every = print_every or self.print_every
        self.plot_every = plot_every or self.plot_every
        if self.plot_every > 0 and self.plotter is None:
            raise ValueError('If plot_every is > 0, a plotter must be passed to the trainer on initialization.')

        if self.plot_every > 0:
            # Fix latents to see how image generation improves during training
            self._fixed_latents = self.G.sample_latent(64)
            if self.use_cuda:
                self._fixed_latents = self._fixed_latents.cuda()
            self._plot_training_sample(0)  # initial sample with untrained models

            # Print header using the keys of the losses dictionary
            header_template = '{:<10}' * (len(self.losses.keys()) + 1)
            tqdm.write(header_template.format('Epoch', *list(self.losses.keys())))

            # template for printing losses during the training process
            print_template = '{:<10}' + '{:<10.4f}' * len(self.losses.keys())

        for epoch in trange(epochs):  # tqdm gives us a nice progress bar
            self._train_epoch(data_loader)
            if self.print_every > 0 and (epoch + 1) % self.print_every == 0:
                tqdm.write(print_template.format(epoch + 1, *[loss[-1] for loss in self.losses.values()]))

            # Save progress
            if self.plot_every > 0 and (epoch + 1) % self.plot_every == 0:
                self._plot_training_sample(epoch + 1)

            # Check early stopping
            if self.check_stopping and self.early_stopping.stop(self.losses):
                tqdm.write('Early stopping at epoch {}'.format(epoch + 1))
                tqdm.write(self.early_stopping.message)
                break

    def _plot_training_sample(self, epoch):
        train_sample = self.G.transformed_sample(self._fixed_latents)
        self.plotter.update(train_sample, self.losses, {'epoch': epoch})

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def save_training_gif(self, filename, duration=5):
        self.plotter.save_gif(filename, duration)
