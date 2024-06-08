import os
import io
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


class TrainingPlotter:
    """ Plots a GIF of the training progress. """
    def __init__(self, loss_names, varnames: list = None, transformer=None) -> None:
        if varnames is None:
            varnames = ['Samples']
        elif isinstance(varnames, str):
            varnames = [varnames]

        self._samples_names = varnames.copy()
        self._loss_names = loss_names.copy()

        self.transformer = transformer

        if len(varnames) == 1:
            varnames = varnames * len(loss_names)
        elif len(varnames) > len(loss_names):
            loss_names += ['.'] * (len(varnames) - len(loss_names))
        elif len(varnames) < len(loss_names):
            varnames += ['.'] * (len(loss_names) - len(varnames))
        self._mosaic = list(zip(varnames, loss_names))

        self.frames = []
        plt.ioff()

    def update(self, samples, losses, meta, save_frame=False):
        """
        Update the plot with the given samples and losses.
        :param samples: a list of samples from the generator
        :param losses: a dictionary of losses
        :param meta: a dictionary of metadata for the plot
        :param save_frame: whether to save the still frame
        """
        fig, ax = plt.subplot_mosaic(self._mosaic,
                                     gridspec_kw={'width_ratios': meta.get('width_ratios', [2, 1])},
                                     figsize=meta.get('figsize', (12, 5)))

        ax[self._samples_names[0]].set_title(f'Samples: Epoch {meta.get("epoch", len(self.frames) + 1)}')
        for i, varname in enumerate(self._samples_names):
            for sample in samples:
                ax[varname].plot(sample[i])
            ax[varname].set_ylabel(varname)

        for j, lossname in enumerate(self._loss_names):
            if lossname not in losses:
                continue
            ax[lossname].set_title(lossname)
            ax[lossname].plot(losses[lossname])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.frames.append(image)
        if save_frame:
            image.save(f'./training_progress/{meta.get("epoch", len(self.frames) + 1)}.png')
        plt.close(fig)

    def save_gif(self, filename, duration=5):
        frame_duration = int(duration * 1000 // len(self.frames))
        imageio.mimsave(filename, self.frames, duration=frame_duration)

    def save_frames(self, filename, save_every=1):
        if filename.endswith('.png'):
            filename = filename[:-4]

        # Make sure the target directory exists
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # save the first and last frames
        self.frames[0].save(f'{filename}_initial.png')
        self.frames[-1].save(f'{filename}_final.png')

        # save intermediate frames
        for i, frame in enumerate(self.frames[1:-1]):  # first and last frames handled separately
            if (i + 1) % save_every == 0:
                frame.save(f'{filename}_{i}.png')


class SDETrainingPlotter(TrainingPlotter):
    """ Plots a GIF of the training progress. """
    def __init__(self, loss_names, varnames: list = None, transformer=None) -> None:
        super().__init__(loss_names, varnames, transformer)

    def update(self, samples, losses, meta, save_frame=False):
        """
        Update the plot with the given samples and losses.
        :param samples: a list of samples from the generator
        :param losses: a dictionary of losses
        :param meta: a dictionary of metadata for the plot
        :param save_frame: whether to save the still frame
        """
        samples = self.transformer.inverse_transform(samples).detach().cpu().numpy()

        fig, ax = plt.subplot_mosaic(self._mosaic,
                                     gridspec_kw={'width_ratios': meta.get('width_ratios', [2, 1])},
                                     figsize=meta.get('figsize', (12, 5)))

        ax[self._samples_names[0]].set_title(f'Samples: Epoch {meta.get("epoch", len(self.frames) + 1)}')
        t = np.arange(samples.shape[1])
        for i, varname in enumerate(self._samples_names):
            for sample in samples:
                ax[varname].plot(t, sample[:, i])  # first row is time
            ax[varname].set_ylabel(varname)

        for j, lossname in enumerate(self._loss_names):
            if lossname not in losses:
                continue
            ax[lossname].set_title(lossname)
            ax[lossname].plot(losses[lossname])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.frames.append(image)
        if save_frame:
            image.save(f'./training_progress/{meta.get("epoch", len(self.frames) + 1)}.png')
        plt.close(fig)
