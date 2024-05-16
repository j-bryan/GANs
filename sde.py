import os
import json
from datetime import datetime

import torch
from torch.optim import Adam
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import make_column_transformer

from models.sde import Generator, Discriminator
from training import WGANClipTrainer, WGANGPTrainer, WGANLPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device


def train_sdegan(params_file: str = None,
                 warm_start: bool = False,
                 epochs: int = 100,
                 device: str | None = None,
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
    if params_file is not None:
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'time_series_length':  24,  # number of nodes in generator output, discriminator input
            'ISO':            'ERCOT',
            'variables':     ['WIND'],
            'time_features':  ['HOD'],
            'initial_noise_size':   4,
            'gen_noise_size':       8,
            'gen_hidden_size':      4,
            'gen_mlp_size':        64,
            'gen_num_layers':       1,
            'dis_hidden_size':      4,
            'dis_mlp_size':        64,
            'dis_num_layers':       1,
            'critic_iterations':   10,
            'batch_size':        1826,
            'gen_lr':            1e-4,
            'dis_lr':            1e-4,
            'gen_betas':   (0.5, 0.9),
            'dis_betas':   (0.5, 0.9),
            'weight_clip':        0.1,
            'epochs':          epochs,
            'total_epochs_trained': 0,
            'random_seed':      12345
        }

    # Find the most appropriate device for training
    device = device or get_accelerator_device()

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]
    # seed for reproducibility
    torch.manual_seed(params['random_seed'])

    G = Generator(data_size=len(params['variables']),
                  initial_noise_size=params['initial_noise_size'],
                  noise_size=params['gen_noise_size'],
                  hidden_size=params['gen_hidden_size'],
                  mlp_size=params['gen_mlp_size'],
                  num_layers=params['gen_num_layers'],
                  time_steps=params['time_series_length']).to(device)
    D = Discriminator(data_size=len(params['variables']),
                      hidden_size=params['dis_hidden_size'],
                      mlp_size=params['dis_mlp_size'],
                      num_layers=params['dis_num_layers']).to(device)
    dataloader, pipeline = get_sde_dataloader(iso=params['ISO'],
                                              varname=params['variables'],
                                              segment_size=params['time_series_length'],
                                              time_features=params['time_features'],
                                              batch_size=params['batch_size'],)
    if params['critic_iterations'] > len(dataloader):
        params['critic_iterations'] = len(dataloader)
        print('Critic iterations reduced to number of batches in dataset:', params['critic_iterations'])

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

    plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'])
    # trainer = WGANClipTrainer(G, D, optimizer_G, optimizer_D,
    #                           weight_clip=params['weight_clip'],
    #                           critic_iterations=params['critic_iterations'],
    #                           plotter=plotter,
    #                           device=device)
    # trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
    #                         plotter=plotter,
    #                         device=device)
    trainer = WGANLPTrainer(G, D, optimizer_G, optimizer_D,
                            plotter=plotter,
                            device=device)

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
    trainer.save_training_gif(dirname + f'training_sde_{iso}_{varnames_abbrev}.gif')
    # Saving individual frames from the GIF. We need to be careful to not save a ton of frames.
    save_every = len(plotter.frames) // 20 + 1  # will save at most 20 frames
    plotter.save_frames(dirname + f'training_progress/training_sde_{iso}_{varnames_abbrev}.png',
                        save_every=save_every)

    # Save models
    torch.save(G.state_dict(), dirname + f'sde_gen_{iso}_{varnames_abbrev}.pt')
    torch.save(D.state_dict(), dirname + f'sde_dis_{iso}_{varnames_abbrev}.pt')

    # Save parameters
    params['total_epochs_trained'] += params['epochs']
    params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # reuse the params_file name if it was specified, otherwise use the default naming scheme
    filename = params_file if params_file is not None else dirname + f'params_sde_{iso}_{varnames_abbrev}.json'
    with open(filename, 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-file', type=str,
                        help='path to a JSON file containing the training parameters for the model')
    parser.add_argument('--warm-start', action='store_true', default=False,
                        help='load saved models and continue training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--device', type=str,
                        help='device to train on')
    parser.add_argument('--no-save', action='store_true', default=False,
                        help='do not save the trained model')
    args = parser.parse_args()

    # NOTE: Instead of specifying a model in the arguments for the warm start case, we rely on the naming
    # convention for the saved models and construct the model name from the parameters specified in
    # the parameters file.

    train_sdegan(params_file=args.params_file,
                 warm_start=args.warm_start,
                 epochs=args.epochs,
                 device=args.device,
                 no_save=args.no_save)
