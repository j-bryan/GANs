from collections import defaultdict
from tqdm import trange, tqdm
import torch
from torch.autograd import Variable
from torch.optim.swa_utils import AveragedModel, SWALR

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
                 generator_iterations: int = 1,
                 plotter: TrainingPlotter | None = None,
                 device: str | None = None,
                 silent: bool = False,
                 swa: bool = True) -> None:
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
        silent: bool, optional (Default: False)
            If True, do not print training progress.
        swa: bool, optional (Default: True)
            If True, use Stochastic Weight Averaging (SWA) for the generator and discriminator. SWA
            will start after half of the training epochs have passed.
        """
        self.device = get_accelerator_device() if device is None else device

        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.G_opt = g_optimizer
        self.D_opt = d_optimizer
        self.losses = {}
        self.critic_iterations = critic_iterations
        self.generator_iterations = generator_iterations

        self._fixed_latent = None
        self.print_every = 0  # by default, don't print or plot anything
        self.plot_every = 0
        self.plotter = plotter

        self.silent = silent

        self._swa = swa
        if self._swa:
            self.G_swa = AveragedModel(self.G)
            self.D_swa = AveragedModel(self.D)

        self._swa_start = torch.inf

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
        generated_data, = self.sample_generator(batch_size, time_steps=data.size(1))

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
        # self.losses['D'].append(d_loss.data.item())
        self.D_opt.step()

        return {'D': d_loss.data.item()}

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
        time_steps = 24 * torch.randint(1, 8, (1,)).item()
        generated_data = self.sample_generator(batch_size, time_steps=time_steps)

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

        # Record loss
        # self.losses['G'].append(g_loss.data.item())
        return {'G': g_loss.data.item()}

    def _train_epoch(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Train the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader for the training data.
        """
        g_losses = defaultdict(float)
        d_losses = defaultdict(float)

        for data in data_loader:
            for i in range(self.critic_iterations):
                critic_losses = self._critic_train_iteration(data)
                for k, v in critic_losses.items():
                    d_losses[k] += v

            for i in range(self.generator_iterations):
                generator_losses = self._generator_train_iteration(data.size()[0])
                for k, v in generator_losses.items():
                    g_losses[k] += v

        epoch_losses = {}
        for k, v in g_losses.items():
            epoch_losses[k] = v / (len(data_loader) * self.generator_iterations)
        for k, v in d_losses.items():
            epoch_losses[k] = v / (len(data_loader) * self.critic_iterations)

        return epoch_losses

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

        losses = defaultdict(list)

        if self._swa:
            self._swa_start = epochs // 2

        if self.plot_every > 0:
            # Fix latents to see how image generation improves during training
            self._fixed_latents = self.G.sample_latent(64).to(self.device)
            self._plot_training_sample(0, losses)  # initial sample with untrained models

            # template for printing losses during the training process

        for epoch in trange(epochs, disable=self.silent):  # tqdm gives us a nice progress bar
            epoch_losses = self._train_epoch(data_loader)
            for k, v in epoch_losses.items():
                losses[k].append(v)

            if epoch == 0 and not self.silent:
                # Print header using the keys of the losses dictionary
                header_template = '{:<10}' * (len(losses.keys()) + 1)
                print_template = '{:<10}' + '{:<10.4f}' * len(losses.keys())
                loss_keys = list(losses.keys())
                tqdm.write(header_template.format('Epoch', *loss_keys))

            if self._swa and epoch >= self._swa_start:
                self.G_swa.update_parameters(self.G)
                self.D_swa.update_parameters(self.D)

            if self.print_every > 0 and (epoch + 1) % self.print_every == 0 and not self.silent:
                tqdm.write(print_template.format(epoch + 1, *[losses[k][-1] for k in loss_keys]))

            # Save progress
            if self.plot_every > 0 and (epoch + 1) % self.plot_every == 0:
                save_frame_counter += 1
                save_frame = save_frame_counter % save_frame_every == 0
                self._plot_training_sample(epoch + 1, losses, save_frame=save_frame)

        self.losses = losses

    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> dict:
        """
        Evaluate the model on a data loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader for the evaluation data.

        Returns
        -------
        losses : dict
            The losses on the evaluation data.
        """
        self.G.eval()
        self.D.eval()

        losses = defaultdict(float)
        n_batches = 0
        for data in data_loader:
            batch_size = data.size()[0]
            latent = self.G.sample_latent(batch_size).to(self.device)
            generated_data, = self.G(latent)

            data = Variable(data).to(self.device)
            d_real = self.D(data)
            d_generated = self.D(generated_data)

            d_loss = d_generated.mean() - d_real.mean()
            g_loss = -d_generated.mean()

            losses['G'] += g_loss.data.item()
            losses['D'] += d_loss.data.item()

            if self._swa:
                generated_data_swa, = self.G_swa(latent)
                d_real_swa = self.D_swa(data)
                d_generated_swa = self.D_swa(generated_data_swa)
                d_loss_swa = d_generated_swa.mean() - d_real_swa.mean()
                g_loss_swa = -d_generated_swa.mean()

                losses['G_swa'] += g_loss_swa.data.item()
                losses['D_swa'] += d_loss_swa.data.item()

            n_batches += 1

        for k, v in losses.items():
            losses[k] /= n_batches

        return losses

    def _plot_training_sample(self, epoch: int, losses: dict, save_frame: bool = False) -> None:
        """
        Plot a sample of the training data.

        Parameters
        ----------
        epoch : int
            The current epoch.
        """
        train_sample = self.G(self._fixed_latents)
        self.plotter.update(train_sample, losses, {'epoch': epoch}, save_frame=save_frame)

    def sample_generator(self, num_samples: int, time_steps: int | None = None) -> torch.Tensor:
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
        return self.G.sample(num_samples, time_steps=time_steps)

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
