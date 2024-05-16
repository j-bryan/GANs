import os
import json
from datetime import datetime

import torch
from torch.optim import Adam

from models.sde import Generator, Discriminator, DriftMLP, DiffusionMLP, DiscriminatorMLP
from models.layers import MLP
from models.initial_conditions import RandNormInitialCondition, ConstantInitialCondition, DataInitialCondition
from training import WGANClipTrainer, WGANGPTrainer, WGANLPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device


def main():
    """
    Fit an SDE-GAN model to univariate ERCOT wind and solar data.
    """
    iso                      = "ERCOT"
    variables                = ["SOLAR"]
    time_features            = ["HOD"]
    batch_size               = 128
    segment_size             = 24
    device                   = "cpu"

    state_size               = len(variables)
    time_features_size       = len(time_features)

    gen_state_size           = state_size
    gen_drift_mlp_size       = 16
    gen_drift_num_layers     = 1
    gen_diffusion_mlp_size   = 16
    gen_diffusion_num_layers = 1
    gen_noise_size           = 1
    dis_mlp_size             = 16
    dis_num_layers           = 1
    gen_lr                   = 4e-5
    dis_lr                   = 4e-5
    gen_betas                = (0.5, 0.9)
    dis_betas                = (0.5, 0.9)
    penalty_weight           = 10.0
    epochs                   = 100
    critic_iterations        = 5
    random_seed              = 12345

    gen_drift     = DriftMLP(state_size=gen_state_size,
                              mlp_size=gen_drift_mlp_size,
                              num_layers=gen_drift_num_layers,
                              time_features=time_features_size,
                              activation='lipswish',
                              final_activation='identity')
    gen_diffusion = DiffusionMLP(state_size=gen_state_size,
                                 noise_size=gen_noise_size,
                                 mlp_size=gen_diffusion_mlp_size,
                                 num_layers=gen_diffusion_num_layers,
                                 time_features=time_features_size,
                                 activation='lipswish',
                                 final_activation='identity')
    gen_initial_condition = ConstantInitialCondition(value=0.0, output_size=gen_state_size)
    gen_initial_embedding = None
    gen_readout = None

    generator = Generator(drift_func=gen_drift,
                          diffusion_func=gen_diffusion,
                          initial_condition=gen_initial_condition,
                          initial_condition_embedding=gen_initial_embedding,
                          readout=gen_readout,
                          time_steps=24).to(device)

    dis_func  = DiscriminatorMLP(gen_state_size=gen_state_size,
                                 data_state_size=state_size,
                                 mlp_size=dis_mlp_size,
                                 num_layers=dis_num_layers,
                                 time_features=time_features_size,
                                 activation='lipswish',
                                 final_activation='sigmoid')
    dis_initial_embedding = None
    dis_readout = None
    discriminator = Discriminator(dis_func, dis_initial_embedding, dis_readout).to(device)

    torch.manual_seed(random_seed)

    # Find the most appropriate device for training

    # Load the data
    dataloader, pipeline = get_sde_dataloader(iso=iso,
                                              varname=variables,
                                              segment_size=segment_size,
                                              time_features=time_features,
                                              batch_size=batch_size)

    optimizer_G = Adam(generator.parameters(), lr=gen_lr, betas=gen_betas)
    optimizer_D = Adam(discriminator.parameters(), lr=dis_lr, betas=dis_betas)

    plotter = SDETrainingPlotter(['G', 'D'], varnames=variables)
    trainer = WGANLPTrainer(generator, discriminator, optimizer_G, optimizer_D, plotter=plotter, device=device)

    plot_every  = max(1, epochs // 100)
    print_every = max(1, epochs // 30)
    trainer.train(data_loader=dataloader, epochs=epochs, plot_every=plot_every, print_every=print_every)

    # Save the trained models, parameters, and visualizations
    dirname = 'saved_models/sde/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save training visualizations
    varnames_abbrev = ''.join([v.lower()[0] for v in variables])
    trainer.save_training_gif(dirname + f'training_sde_{iso}_{varnames_abbrev}.gif')
    # Saving individual frames from the GIF. We need to be careful to not save a ton of frames.
    save_every = len(plotter.frames) // 20 + 1  # will save at most 20 frames
    plotter.save_frames(dirname + f'training_progress/training_sde_{iso}_{varnames_abbrev}.png',
                        save_every=save_every)

    # Save models
    torch.save(generator.state_dict(), dirname + f'sde_gen_{iso}_{varnames_abbrev}.pt')
    torch.save(discriminator.state_dict(), dirname + f'sde_dis_{iso}_{varnames_abbrev}.pt')


if __name__ == '__main__':
    main()
