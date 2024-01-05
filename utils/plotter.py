import io
from PIL import Image
import imageio
import matplotlib.pyplot as plt


class TrainingPlotter:
    """ Plots a GIF of the training progress. """
    def __init__(self, loss_names, varnames: list = None) -> None:
        if varnames is None:
            varnames = ['Samples']
        elif isinstance(varnames, str):
            varnames = [varnames]

        self._samples_names = varnames.copy()
        self._loss_names = loss_names.copy()

        if len(varnames) == 1:
            varnames = varnames * len(loss_names)
        elif len(varnames) > len(loss_names):
            loss_names += ['.'] * (len(varnames) - len(loss_names))
        elif len(varnames) < len(loss_names):
            varnames += ['.'] * (len(loss_names) - len(varnames))
        self._mosaic = list(zip(varnames, loss_names))

        self.frames = []
        plt.ioff()

    def update(self, samples, losses, meta):
        """
        Update the plot with the given samples and losses.
        :param samples: a list of samples from the generator
        :param losses: a dictionary of losses
        :param meta: a dictionary of metadata for the plot
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
            ax[lossname].set_title(lossname)
            ax[lossname].plot(losses[lossname])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.frames.append(image)
        plt.close(fig)

    def save_gif(self, filename, duration=5):
        frame_duration = int(duration * 1000 // len(self.frames))
        imageio.mimsave(filename, self.frames, duration=frame_duration)

    def save_frames(self, filename, save_every=1):
        if filename.endswith('.png'):
            filename = filename[:-4]

        for i, frame in enumerate(self.frames):
            if i % save_every == 0:
                frame.save(f'{filename}_{i}.png')


class SDETrainingPlotter(TrainingPlotter):
    """ Plots a GIF of the training progress. """
    def __init__(self, loss_names, varnames: list = None) -> None:
        super().__init__(loss_names, varnames)

    def update(self, samples, losses, meta):
        """
        Update the plot with the given samples and losses.
        :param samples: a list of samples from the generator
        :param losses: a dictionary of losses
        :param meta: a dictionary of metadata for the plot
        """
        fig, ax = plt.subplot_mosaic(self._mosaic,
                                     gridspec_kw={'width_ratios': meta.get('width_ratios', [2, 1])},
                                     figsize=meta.get('figsize', (12, 5)))

        ax[self._samples_names[0]].set_title(f'Samples: Epoch {meta.get("epoch", len(self.frames) + 1)}')
        for i, varname in enumerate(self._samples_names):
            for sample in samples:
                ax[varname].plot(sample[0], sample[i + 1])  # first row is time
            ax[varname].set_ylabel(varname)

        for j, lossname in enumerate(self._loss_names):
            ax[lossname].set_title(lossname)
            ax[lossname].plot(losses[lossname])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.frames.append(image)
        plt.close(fig)
