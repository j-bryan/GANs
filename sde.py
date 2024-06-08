import os
import json
from datetime import datetime
import fire

import numpy as np
import torch
from torch.optim import Adam
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import make_column_transformer

from models.sde import Generator, DiscriminatorSimple, SdeGeneratorConfig
from training import WGANClipTrainer, WGANGPTrainer, WGANLPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device

from dataclasses import dataclass
from models.layers import FFNNConfig

from evaluate_sde import plot_model_results


@dataclass
class AdamConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0

    def to_dict(self, prefix: str = ""):
        # prepend the prefix to the keys
        if prefix == "":
            return self.__dict__
        return {prefix + "_" + k: v for k, v in self.__dict__.items()}


def train_sdegan(params_file: str = None,
                 warm_start: bool = False,
                 epochs: int = 100,
                 device: str | None = None,
                 no_save: bool = False,
                 silent: bool = False,
                 hidden_size: int = 16) -> None:
    """
    Sets up and trains an SDE-GAN.

    Parameters
    ----------
    params_file : str, optional
        Path to a JSON file containing the parameters for the model. If None, the parameter set defined
        in the function will be used.
    warm_start : bool, optional
        If True, load saved models and continue training. Requires params_file to be specified.
        If False, train a new model.
    epochs : int, optional
        The number of epochs to train for.
    """
    # Set up training. All of these parameters are saved along with the models so the training can be reproduced.
    if params_file is not None:
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {
            "ISO": "ERCOT",
            "variables": ["TOTALLOAD", "WIND", "SOLAR"],
            "time_features": ["HOD"],
            "time_series_length": 24,
            "critic_iterations": 5,
            "penalty_weight": 10.0,
            "epochs": epochs,
            "total_epochs_trained": 0,
            "random_seed": 12345,
            "batch_size": 64
        }
        data_size = len(params["variables"])
        time_size = len(params["time_features"])
        initial_noise_size = 16
        # hidden_size = 16

        gen_noise_embed_config = FFNNConfig(
            in_size=initial_noise_size,
            num_layers=2,
            num_units=32,
            out_size=hidden_size,
        )
        gen_drift_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=3,
            num_units=64,
            out_size=hidden_size,
            final_activation="tanh"
        )
        gen_diffusion_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=3,
            num_units=64,
            out_size=hidden_size,
            final_activation="tanh"
        )
        gen_readout_config = FFNNConfig(
            in_size=hidden_size+time_size,
            num_layers=3,
            num_units=64,
            out_size=len(params["variables"]),
            final_activation=["identity", "sigmoid", "hardsigmoid"]
        )
        sde_generator_config = SdeGeneratorConfig(
            noise_type="diagonal",
            sde_type="stratonovich",
            time_steps=params["time_series_length"],
            time_size=time_size,
            data_size=3,
            init_noise_size=initial_noise_size,
            noise_size=hidden_size,
            hidden_size=hidden_size,
            drift_config=gen_drift_config,
            diffusion_config=gen_diffusion_config,
            embed_config=gen_noise_embed_config,
            readout_config=gen_readout_config
        )
        discriminator_config = FFNNConfig(
            in_size=data_size * params["time_series_length"],
            num_layers=5,
            num_units=256,
            out_size=1
        )
        gen_opt_config = AdamConfig(
            lr=1e-4,
            betas=(0.5, 0.9),
            weight_decay=0.0
        )
        dis_opt_config = AdamConfig(
            lr=1e-4,
            betas=(0.5, 0.9),
            weight_decay=0.0
        )

        params.update(sde_generator_config.to_dict())
        params.update(discriminator_config.to_dict(prefix="dis"))
        params.update(gen_opt_config.to_dict(prefix="gen"))
        params.update(dis_opt_config.to_dict(prefix="dis"))

    # Find the most appropriate device for training
    device = device or get_accelerator_device()

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]
    # seed for reproducibility
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])

    G = Generator(sde_generator_config).to(device)
    D = DiscriminatorSimple(discriminator_config).to(device)
    dataloader, transformer = get_sde_dataloader(iso=params['ISO'],
                                                 varname=params['variables'],
                                                 segment_size=params['time_series_length'],
                                                 batch_size=params['batch_size'],
                                                 device=device)

    optimizer_G = Adam([
            {"params": G._initial.parameters(), "lr": 5*params["gen_lr"]},
            # {"params": G._initial.parameters()},
            {"params": G._func.parameters()},
            {"params": G._readout.parameters()}
        ], lr=params['gen_lr'], betas=params['gen_betas'])
    optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])

    if warm_start and params_file is not None:
        # load the models based on the model naming scheme for CNN models:
        #    saved_models/cnn/cnn_{gen/dis}_{ISO}_{var1}{var2}...{varn}.pt
        # where {var1}...{varn} are the lowercase first letters of the variable names. This variable
        # naming scheme isn't ideal since there can be collisions, but for the variables we're using
        # it should be fine.
        G.load_state_dict(torch.load(f'saved_models/sde/sde_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
        D.load_state_dict(torch.load(f'saved_models/sde/sde_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'], transformer=transformer)
    trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                            critic_iterations=params['critic_iterations'],
                            plotter=plotter,
                            device=device,
                            silent=silent,
                            swa=True)

    plot_every  = max(1, params['epochs'] // 100)
    print_every = max(1, params['epochs'] // 30)
    trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=plot_every, print_every=print_every)

    if no_save:
        return

    # Save the trained models, parameters, and visualizations
    dirname = 'saved_models/sde/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save training visualizations
    iso = params['ISO']
    postfix = f"hiddensize{hidden_size}_readoutHour"
    varnames_abbrev = ''.join([v.lower()[0] for v in params['variables']])
    trainer.save_training_gif(dirname + f'training_sde_{iso}_{varnames_abbrev}_{postfix}.gif')
    # Saving individual frames from the GIF. We need to be careful to not save a ton of frames.
    save_every = len(plotter.frames) // 20 + 1  # will save at most 20 frames
    plotter.save_frames(dirname + f'training_progress/training_sde_{iso}_{varnames_abbrev}_{postfix}.png',
                        save_every=save_every)

    # Save models
    torch.save(G.state_dict(), dirname + f'sde_gen_{iso}_{varnames_abbrev}_{postfix}.pt')
    torch.save(D.state_dict(), dirname + f'sde_dis_{iso}_{varnames_abbrev}_{postfix}.pt')

    if trainer._swa:
        torch.save(trainer.G_swa.state_dict(), dirname + f'sde_gen_swa_{iso}_{varnames_abbrev}_{postfix}.pt')
        torch.save(trainer.D_swa.state_dict(), dirname + f'sde_dis_swa_{iso}_{varnames_abbrev}_{postfix}.pt')

    # Save parameters
    params['total_epochs_trained'] += params['epochs']
    params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # reuse the params_file name if it was specified, otherwise use the default naming scheme
    filename = params_file if params_file is not None else dirname + f'params_sde_{iso}_{varnames_abbrev}_{postfix}.json'
    with open(filename, 'w') as f:
        json.dump(params, f)

    plot_model_results(G, transformer, params["variables"],
                       G_swa=trainer.G_swa if trainer._swa else None,
                       dir_suffix=f"ep{params['epochs']}_hidden{params['hidden_size']}_swa")


if __name__ == '__main__':
    fire.Fire(train_sdegan)
