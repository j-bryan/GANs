from tqdm import trange, tqdm
import torch
from torch.autograd import Variable

from utils.plotting import TrainingPlotter
from utils import get_accelerator_device


class Trainer:
    """
    Basic WGAN training. Variations to the loss function, such as gradient penalty and
    weight clipping, are implemented in their own child classes.
    """
    def __init__(self,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
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
        critic_iterations : int
            The number of iterations to train the critic for each generator iteration.
        plotter : TrainingPlotter, optional
            The plotter to use for plotting training progress.
        device : str, optional
            The device to use for training. If None, a GPU is used if available, otherwise
            defaulting to CPU.
        """
        self.device = get_accelerator_device() if device is None else device

        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.G_opt = g_optimizer
        self.D_opt = d_optimizer
        self.losses = {'G': [], 'D': []}
        self.num_steps = 0
        self.critic_iterations = critic_iterations

        self._fixed_latent = None
        self.print_every = 0  # by default, don't print or plot anything
        self.plot_every = 0
        self.plotter = plotter

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
        generated_data, = self.sample_generator(batch_size)

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

    def _generator_train_iteration(self, batch_size: int) -> None:
        """
        Train the generator for one iteration.

        Parameters
        ----------
        batch_size : int
            Batch size for generated data.
        """
        self.G_opt.zero_grad()

        # Get generated data
        # generated_data = self.sample_generator(batch_size)
        # generated_data, f_periodicity_diff, g_periodicity_diff = self.sample_generator(batch_size, t_offset=24, return_fg=True)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        # The generator outputs a number between 0 and 1, corresponding to the probability that the sample is real.
        # If the generator is performing well, the discriminator is fooled by the generated data and produces a high value.
        # We want to maximize that, so we flip the sign.
        d_generated = self.D(generated_data)
        # We want to enforce periodicity (period of 24) in the drift and diffusion functions of G.
        g_loss = -d_generated.mean()
        # g_loss = -d_generated.mean() + torch.norm(f_periodicity_diff, 2) + torch.norm(g_periodicity_diff, 2)
        g_loss.backward()
        self.G_opt.step()

        # Print model parameters to check if updates for initial value embedding is actually happening
        # for name, param in self.G._readout.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     print(name, param.grad.view(-1))
        #     # print(name, param.data)

        # Record loss
        self.losses['G'].append(g_loss.data.item())

    def _train_epoch(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Train the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader for the training data.
        """
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            if i % self.critic_iterations == 0:  # only update generator every critic_iterations iterations
                self._generator_train_iteration(data.size()[0])
            # for i in range(self.critic_iterations):
            #     self._critic_train_iteration(data)
            # self._generator_train_iteration(data.size()[0])

    def train(self,
              data_loader: torch.utils.data.DataLoader,
              epochs: int,
              print_every: int | None = None,
              plot_every: int | None = None):
        """
        Train the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader for the training data.
        epochs : int
            The number of epochs to train for.
        print_every : int, optional
            If not None, print losses every `print_every` epochs.
        plot_every : int, optional
            If not None, plot a sample every `plot_every` epochs. Requires a plotter to be given on initialization.
        """
        self.print_every = print_every or self.print_every
        self.plot_every = plot_every or self.plot_every
        if self.plot_every > 0 and self.plotter is None:
            raise ValueError('If plot_every is > 0, a plotter must be passed to the trainer on initialization.')
        # We only want to save more frames in the gif than we do stills, but we still want
        # to save some stills during the training process so we can see how the model is improving.
        # We only want to save ~30 still frames during training. For every time we add a frame to the
        # gif, we need to decide if we should save a still frame.
        gif_n_frames = epochs // self.plot_every  # number of frames in the gif
        save_frame_every = max(1, gif_n_frames // 30)  # save a frame every 30 frames
        save_frame_counter = 0

        if self.plot_every > 0:
            # Fix latents to see how image generation improves during training
            self._fixed_latents = self.G.sample_latent(64).to(self.device)
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
                save_frame_counter += 1
                save_frame = save_frame_counter % save_frame_every == 0
                self._plot_training_sample(epoch + 1, save_frame=save_frame)

    def _plot_training_sample(self, epoch: int, save_frame: bool = False) -> None:
        """
        Plot a sample of the training data.

        Parameters
        ----------
        epoch : int
            The current epoch.
        """
        train_sample = self.G(self._fixed_latents)
        self.plotter.update(train_sample, self.losses, {'epoch': epoch}, save_frame=save_frame)

    def sample_generator(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the generator.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.

        Returns
        -------
        generated_data : torch.Tensor
            The generated data.
        """
        return self.G.sample(num_samples)

    def save_training_gif(self, filename: str, duration: int = 5) -> None:
        """
        Save a GIF of the training progress.

        Parameters
        ----------
        filename : str
            The path to save the GIF to.
        duration : int, optional
            The approximate duration of the GIF in seconds.
        """
        self.plotter.save_gif(filename, duration)
