import json
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


def train_cnn(warm_start: bool = False):
    # Set up training. All of these parameters are saved along with the models so the training can be reproduced.
    params = {'time_series_length': 24,  # number of nodes in generator output, discriminator input
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
              'epochs': 500,
              'random_seed': 12345}

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
    if torch.cuda.is_available():
        G.cuda().single()
        D.cuda().single()
        G_init_input = G_init_input.cuda()
        D_init_input = D_init_input.cuda()
    else:
        G.cpu()
        D.cpu()
        G_init_input = G_init_input.cpu()
        D_init_input = D_init_input.cpu()
    G(G_init_input)
    D(D_init_input)

    if warm_start:
        # load the models
        G.load_state_dict(torch.load(f'saved_models/cnn_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
        D.load_state_dict(torch.load(f'saved_models/cnn_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    optimizer_G = Adam(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
    optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])

    plotter = TrainingPlotter(['G', 'D'], varnames=params['variables'])
    trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                            gp_weight=params['gp_weight'],
                            critic_iterations=params['critic_iterations'],
                            plotter=plotter)

    trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=10, print_every=10)
    trainer.save_training_gif('training.gif')

    # Save models and hyperparameters
    torch.save(G.state_dict(), f'saved_models/cnn_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    torch.save(D.state_dict(), f'saved_models/cnn_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    with open(f'saved_models/cnn_params_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.json', 'w') as f:
        json.dump(params, f)


def evaluate_cnn():
    with open('saved_models/cnn_params_ERCOT_sw.json', 'r') as f:
        params = json.load(f)

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

    dataloader, pipeline = get_dataloader(iso=params['ISO'],
                                          varname=params['variables'],
                                          segment_size=params['time_series_length'],
                                          batch_size=params['batch_size'])

    # Since the Generator and Discriminator use lazy layer initialization, we need to move them to the correct device,
    # specify data types, and call them once to initialize the layers.
    G_init_input = torch.ones((1, params['gen_input_size']))
    D_init_input = torch.ones((1, len(params['variables']), params['time_series_length']))
    if torch.cuda.is_available():
        G.cuda().single()
        D.cuda().single()
        G_init_input = G_init_input.cuda()
        D_init_input = D_init_input.cuda()
    else:
        G.cpu()
        D.cpu()
        G_init_input = G_init_input.cpu()
        D_init_input = D_init_input.cpu()
    G(G_init_input)
    D(D_init_input)

    # load the models
    G.load_state_dict(torch.load(f'saved_models/cnn_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
    D.load_state_dict(torch.load(f'saved_models/cnn_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    # Evaluate the model for basic statistical properties
    # First, we need to sample the generator. We'll use 1000 samples.
    latent_samples = G.sample_latent(1000)
    synthetic = G.transformed_sample(latent_samples)
    historical = pipeline.inverse_transform(dataloader.dataset.data.T).T
    historical = np.array(np.split(historical, historical.shape[1] // params['time_series_length'], axis=1))

    # hourly_distributions(synthetic, historical)
    autocorrelation(synthetic, historical, params['variables'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    if args.warm_start:
        train_cnn(warm_start=args.warm_start)
    elif args.evaluate:
        evaluate_cnn()
