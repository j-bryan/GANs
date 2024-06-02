import os
import json
from datetime import datetime
import fire
import numpy as np

import torch
from torch.optim import Adam

from models.forcing import DawnDuskSampler
from models.sde_no_embed import SampledInitialCondition
from models.sde_statespace import Generator, DiscriminatorSimple
from training import WGANGPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device


def train_sdegan(params_file: str = "",
                 warm_start: bool = False,
                 epochs: int = 100,
                 device: str = "cpu",
                 no_save: bool = False) -> None:
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
    if params_file:
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'time_series_length':  24,  # number of nodes in generator output, discriminator input
            'ISO':            'ERCOT',
            'variables':     ['TOTALLOAD', 'WIND', 'SOLAR'],
            'aux_ou':           False,
            'noise_size':           3,
            'gen_f_num_units':    128,
            'gen_f_num_layers':     3,
            'gen_g_num_units':    128,
            'gen_g_num_layers':     3,
            'aux_size':             3,
            'dis_num_units':      256,
            'dis_num_layers':       5,
            'critic_iterations':   10,
            'batch_size':        1826,
            'gen_lr':            1e-5,
            'dis_lr':            2e-5,
            'gen_betas':   (0.5, 0.9),
            'dis_betas':   (0.5, 0.9),
            'gp_weight':         10.0,
            'epochs':          epochs,
            'total_epochs_trained': 0,
            'random_seed':      12345
        }

    # Find the most appropriate device for training
    device = device or get_accelerator_device()

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]
    # seed for reproducibility
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])

    dataloader, transformer = get_sde_dataloader(iso=params['ISO'],
                                                 varname=params['variables'],
                                                 segment_size=params['time_series_length'],
                                                 batch_size=params['batch_size'],
                                                 device=device)

    # Some of the data is bad! Solar values must be 0 at night. We'll drop the bad rows.
    solar = dataloader.dataset[..., params["variables"].index("SOLAR")]
    solar[:, [0, 1, 2, 3, 4, 21, 22, 23]] = 0.0
    dataset = dataloader.dataset
    dataset[..., params["variables"].index("SOLAR")] = solar
    # Remake the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    params["batch_size"] = len(dataloader)
    dawn_dusk_sampler = DawnDuskSampler(solar)

    # The initial condition for the main state variables are sampled directly from the data.
    initial_condition = SampledInitialCondition(data=dataloader.dataset)
    state_size = len(params['variables'])

    G = Generator(initial_condition=initial_condition,
                  time_size=params['time_series_length'],
                  state_size=state_size,
                  noise_size=params['noise_size'],
                  f_num_units=params['gen_f_num_units'],
                  f_num_layers=params['gen_f_num_layers'],
                  g_num_units=params['gen_g_num_units'],
                  g_num_layers=params['gen_g_num_layers'],
                  aux_size=params['aux_size'],
                  varnames=params['variables'],
                  aux_ou=params['aux_ou'],
                  dawn_dusk_sampler=dawn_dusk_sampler).to(device)
    D = DiscriminatorSimple(data_size=state_size,
                            time_size=params['time_series_length'],
                            num_layers=params['dis_num_layers'],
                            num_units=params['dis_num_units']).to(device)

    optimizer_G = Adam(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
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
                            plotter=plotter,
                            device=device,
                            penalty_weight=params['gp_weight'])

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
    varnames_abbrev = ''.join([v.lower()[0] for v in params['variables']])
    trainer.save_training_gif(dirname + f'training_sde_noembed_{iso}_{varnames_abbrev}.gif')
    # Saving individual frames from the GIF. We need to be careful to not save a ton of frames.
    save_every = len(plotter.frames) // 20 + 1  # will save at most 20 frames
    plotter.save_frames(dirname + f'training_progress/training_sde_statespace_{iso}_{varnames_abbrev}.png',
                        save_every=save_every)

    # Save models
    torch.save(G.state_dict(), dirname + f'sde_gen_{iso}_{varnames_abbrev}.pt')
    torch.save(D.state_dict(), dirname + f'sde_dis_{iso}_{varnames_abbrev}.pt')

    # Save parameters
    params['total_epochs_trained'] += params['epochs']
    params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # reuse the params_file name if it was specified, otherwise use the default naming scheme
    filename = params_file if params_file else dirname + f'params_sde_{iso}_{varnames_abbrev}.json'
    with open(filename, 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    fire.Fire(train_sdegan)
