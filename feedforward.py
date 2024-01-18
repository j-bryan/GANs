import json
import torch
from torch.optim import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from models.feedforward import Generator, Discriminator
from models.preprocessing import ManualMinMaxScaler
from training import WGANGPTrainer
from dataloaders import get_dataloader
from utils import TrainingPlotter


def train_feedfoward():
    # Set up training. All of these parameters are saved along with the models so the training can be reproduced.
    params = {'time_series_length': 24,  # number of nodes in generator output, discriminator input
              'ISO': 'ERCOT',
              'variables': ['SOLAR'],
              'gen_input_size': 100,
              'gen_hidden_size': 256,  # number of nodes
              'gen_num_layers': 3,     # number of layers
              'dis_hidden_size': 256,
              'dis_num_layers': 3,
              'gp_weight': 10,
              'critic_iterations': 10,
              'batch_size': 32,
              'gen_lr': 1e-4,
              'dis_lr': 1e-4,
              'gen_betas': (0.5, 0.9),
              'dis_betas': (0.5, 0.9),
              'epochs': 30,
              'random_seed': 12345}

    if isinstance(params['variables'], str):
        params['variables'] = [params['variables']]
    torch.manual_seed(params['random_seed'])

    G = Generator(input_size=params['gen_input_size'],
                  hidden_size=params['gen_hidden_size'],
                  num_layers=params['gen_num_layers'],
                  output_size=params['time_series_length'],
                  num_vars=len(params['variables']))
    D = Discriminator(input_size=params['time_series_length'],
                      hidden_size=params['dis_hidden_size'],
                      num_layers=params['dis_num_layers'],
                      num_vars=len(params['variables']))

    preprocessor = make_pipeline(
        make_column_transformer(
            (ManualMinMaxScaler((0, 1), (-1, 1)), [params['variables'].index('WIND'), params['variables'].index('SOLAR')]),
            (RobustScaler(), params['variables'].index('TOTALLOAD'))
        ),
        'passthrough'
    )
    dataloader, pipeline = get_dataloader(iso=params['ISO'],
                                          varname=params['variables'],
                                          segment_size=params['time_series_length'],
                                          batch_size=params['batch_size'],
                                          preprocessor=preprocessor)
    G.preprocessor = pipeline  # save the preprocessor with the model so it can be used later

    optimizer_G = Adam(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
    optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])

    plotter = TrainingPlotter(['G', 'D'], varnames=params['variables'])
    trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                            gp_weight=params['gp_weight'],
                            critic_iterations=params['critic_iterations'],
                            plotter=plotter)

    trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=1, print_every=1)
    trainer.save_training_gif('training.gif')

    # Save models and hyperparameters
    torch.save(G.state_dict(), f'saved_models/ff_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    torch.save(D.state_dict(), f'saved_models/ff_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    with open(f'saved_models/ff_params_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.json', 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    train_feedfoward()
