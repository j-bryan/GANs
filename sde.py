import json
import torch
from torch.optim import Adam
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import make_column_transformer

from models.sde import Generator, Discriminator
from training import SDEGANTrainer
from dataloaders import get_sde_dataloader
from utils import SDETrainingPlotter


def train_sdegan():
    # Set up training. All of these parameters are saved along with the models so the training can be reproduced.
    params = {'time_series_length': 24,  # number of nodes in generator output, discriminator input
              'ISO': 'ERCOT',
              'variables': ['WIND'],
              'time_features': ['HOD'],
              'initial_noise_size': 100,
              'gen_noise_size':       4,
              'gen_hidden_size':      4,
              'gen_mlp_size':        64,
              'gen_num_layers':       3,
              'dis_hidden_size':      4,
              'dis_mlp_size':       128,
              'dis_num_layers':       3,
              'critic_iterations':   10,
              'batch_size':          32,
              'gen_lr':            1e-5,
              'dis_lr':            1e-5,
              'gen_betas':   (0.5, 0.9),
              'dis_betas':   (0.5, 0.9),
              'weight_clip':       0.03,
              'epochs':              10,
              'random_seed':      12345}

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
                  time_steps=params['time_series_length'])
    D = Discriminator(data_size=len(params['variables']),
                      hidden_size=params['dis_hidden_size'],
                      mlp_size=params['dis_mlp_size'],
                      num_layers=params['dis_num_layers'])
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

    plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'])
    trainer = SDEGANTrainer(G, D, optimizer_G, optimizer_D,
                            weight_clip=params['weight_clip'],
                            critic_iterations=params['critic_iterations'],
                            plotter=plotter,
                            # use_cuda=torch.cuda.is_available())
                            use_cuda=False)

    trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=1, print_every=1)
    trainer.save_training_gif('training.gif')

    # Save models and hyperparameters
    torch.save(G.state_dict(), f'saved_models/sde_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    torch.save(D.state_dict(), f'saved_models/sde_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt')
    with open(f'saved_models/sde_params_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.json', 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    train_sdegan()
