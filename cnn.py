import json
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from models.conv import Generator, Discriminator
from models.preprocessing import ManualMinMaxScaler
from training import WGANGPTrainer
from dataloaders import get_dataloader
from utils import TrainingPlotter
from scoring import hourly_distributions, autocorrelation


def train_cnn(params_file: str = None, warm_start: bool = False, epochs: int = 100) -> None:
    """
    Sets up and trains a convolutional GAN.

    Parameters
    ----------
    params_file : str, optional
        Path to a JSON file containing the parameters for the model. If None, the parameter set defined
        in the function will be used.
    warm_start : bool, optional
        If True, load saved models and continue training. Requires params_file to be specified.
        If False, train a new model.
    """
    # Set up training. All of these parameters are saved along with the models so the training can be reproduced.
    if params_file is not None:
        with open(params_file, 'r') as f:
            params = json.load(f)
        params['epochs'] = epochs  # can be set from command line argument --epochs
    else:
        params = {
            'time_series_length': 24,  # number of nodes in generator output, discriminator input
            'ISO': 'ERCOT',
            'variables': ['SOLAR', 'WIND'],
            'gen_input_size': 100,
            'gen_num_filters': 12,
            'gen_num_layers': 3,
            'dis_num_filters': 12,
            'dis_num_layers': 3,
            'gp_weight': 10,
            'critic_iterations': 10,
            'batch_size': 32,
            'gen_lr': 1e-4,
            'dis_lr': 1e-4,
            'gen_betas': (0.5, 0.9),
            'dis_betas': (0.5, 0.9),
            'epochs': epochs,  # moved to function/command line argument to accommodate warm start
            'total_epochs_trained': 0,  # to keep track of how many epochs have been trained in case of warm start
            'random_seed': 12345
        }

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]
    torch.manual_seed(params['random_seed'])

    G = Generator(input_size=params['gen_input_size'],
                  num_filters=params['gen_num_filters'],
                  num_layers=params['gen_num_layers'],
                  output_size=params['time_series_length'],
                  num_vars=len(params['variables']))
    D = Discriminator(num_filters=params['dis_num_filters'],
                      num_layers=params['dis_num_layers'])
    # preprocessor = make_pipeline(
    #     make_column_transformer(
    #         (ManualMinMaxScaler((0, 1), (-1, 1)), [params['variables'].index('WIND'), params['variables'].index('SOLAR')]),
    #         (RobustScaler(), params['variables'].index('TOTALLOAD'))
    #     ),
    #     'passthrough'
    # )
    dataloader, pipeline = get_dataloader(iso=params['ISO'],
                                          varname=params['variables'],
                                          segment_size=params['time_series_length'],
                                          batch_size=params['batch_size'])
                                        #   preprocessor=preprocessor)
    # G.preprocessor = pipeline  # save the preprocessor with the model so it can be used later

    # Since the Generator and Discriminator use lazy layer initialization, we need to move them to the correct device,
    # specify data types, and call them once to initialize the layers.
    G_init_input = torch.ones((1, params['gen_input_size']))
    D_init_input = torch.ones((1, len(params['variables']), params['time_series_length']))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        G.cuda()
        D.cuda()
        G_init_input = G_init_input.cuda()
        D_init_input = D_init_input.cuda()
    else:
        G.cpu()
        D.cpu()
        G_init_input = G_init_input.cpu()
        D_init_input = D_init_input.cpu()
    G(G_init_input)
    D(D_init_input)

    # For now, we'll require params_file to be specified for warm start.
    if warm_start and params_file is not None:
        # load the models based on the model naming scheme for CNN models:
        #    saved_models/cnn/cnn_{gen/dis}_{ISO}_{var1}{var2}...{varn}.pt
        # where {var1}...{varn} are the lowercase first letters of the variable names. This variable
        # naming scheme isn't ideal since there can be collisions, but for the variables we're using
        # it should be fine.
        G.load_state_dict(torch.load(f'saved_models/cnn/cnn_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
        D.load_state_dict(torch.load(f'saved_models/cnn/cnn_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    optimizer_G = Adam(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
    optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])

    plotter = TrainingPlotter(['G', 'D'], varnames=params['variables'])
    trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                            gp_weight=params['gp_weight'],
                            critic_iterations=params['critic_iterations'],
                            plotter=plotter,
                            use_cuda=use_cuda)

    # Let's try to be smart about the frequency we print and plot. This should be proportional to the
    # number of epochs we're training for. Let's aim for up to 100 of each.
    plot_every  = max(1, params['epochs'] // 100)
    print_every = max(1, params['epochs'] // 100)
    trainer.train(data_loader=dataloader,
                  epochs=params['epochs'],
                  plot_every=plot_every,
                  print_every=print_every)

    # Save training visualizations
    iso = params['ISO']
    varnames_abbrev = ''.join([v.lower()[0] for v in params['variables']])
    trainer.save_training_gif(f'saved_models/cnn/training_cnn_{iso}_{varnames_abbrev}.gif')
    # Saving individual frames from the GIF. We need to be careful to not save a ton of frames.
    save_every = len(plotter.frames) // 20 + 1  # will save at most 20 frames
    plotter.save_frames(f'saved_models/cnn/training_progress/training_cnn_{iso}_{varnames_abbrev}.png',
                        save_every=save_every)

    # Save models
    torch.save(G.state_dict(), f'saved_models/cnn_gen_{iso}_{varnames_abbrev}.pt')
    torch.save(D.state_dict(), f'saved_models/cnn_dis_{iso}_{varnames_abbrev}.pt')

    # Save parameters
    params['total_epochs_trained'] += params['epochs']
    params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # reuse the params_file name if it was specified, otherwise use the default naming scheme
    filename = params_file if params_file is not None else f'saved_models/cnn/params_cnn_{iso}_{varnames_abbrev}.json'
    with open(filename, 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-file', type=str, help='path to a JSON file containing the training parameters for the model')
    parser.add_argument('--warm-start', action='store_true', default=False, help='load saved models and continue training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    args = parser.parse_args()

    # Instead of specifying a model in the arguments for the warm start case, we rely on the naming
    # convention for the saved models and construct the model name from the parameters specified in
    # the

    train_cnn(params_file=args.params_file,
              warm_start=args.warm_start,
              epochs=args.epochs)
