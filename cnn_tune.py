import os
import json
from datetime import datetime
import fire

import numpy as np
import torch
from torch.optim import Adam
from scipy.stats import wasserstein_distance

from models.conv import Generator, Discriminator
from training import WGANGPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter

from dataclasses import dataclass

from evaluate_sde import plot_model_results, calculate_metrics

import optuna


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


def tune_cnn_gan(n_trials: int = 128,
                 epochs: int = 2000,
                 batch_size: int = 512,
                 noise_size: int = 24,
                 device: str = "cuda",
                 silent: bool = False) -> None:
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
    segment_size = 24
    dataloader, _, _, transformer = get_sde_dataloader(
        iso="ERCOT",
        varname=["TOTALLOAD", "WIND", "SOLAR"],
        segment_size=segment_size,
        batch_size=batch_size,
        device=device,
        test_size=0.0,
        valid_size=0.0,
    )
    critic_iterations = max(1, min(10, len(dataloader)))

    def objective(trial: optuna.Trial, dirname: str | None = None):
        gen_num_filters = trial.suggest_categorical("gen_num_filters", [8, 16, 32, 64, 128])
        gen_num_layers = trial.suggest_int("gen_num_layers", 2, 4)
        dis_num_filters = trial.suggest_categorical("dis_num_filters", [8, 16, 32, 64, 128])
        dis_num_layers = trial.suggest_int("dis_num_layers", 2, 4)
        params = {
            "ISO":                "ERCOT",
            "variables":          ["TOTALLOAD", "WIND", "SOLAR"],
            "time_features":      ["HOD"],
            "time_series_length": segment_size,
            "critic_iterations":  critic_iterations,
            "penalty_weight":     10.0,
            "epochs":             epochs,
            "random_seed":        12345,
            "batch_size":         batch_size,
            # Generator parameters
            "init_noise_size":    noise_size,  # tune?
            "gen_num_filters":    gen_num_filters,
            "gen_num_layers":     gen_num_layers,
            # Discriminator parameters
            "dis_num_filters":    dis_num_filters,
            "dis_num_layers":     dis_num_layers,
        }

        readout_activations = {
            "TOTALLOAD": "identity",    # output in (-inf, inf)
            "WIND":      "sigmoid",     # output in (0, 1)
            "SOLAR":     "hardsigmoid"  # output in [0, 1]
        }

        data_size = len(params["variables"])

        if isinstance(params['variables'], str):
            params['variables'] = [params['variables']]
        # seed for reproducibility
        np.random.seed(params['random_seed'])
        torch.manual_seed(params['random_seed'])

        G = Generator(
            input_size=params["init_noise_size"],
            num_filters=params["gen_num_filters"],
            num_layers=params["gen_num_layers"],
            output_size=params["time_series_length"],
            output_activation=[readout_activations[v] for v in params["variables"]],
            num_vars=data_size
        ).to(device)
        D = Discriminator(
            num_filters=params["dis_num_filters"],
            num_layers=params["dis_num_layers"]
        ).to(device)

        # Since the Generator and Discriminator use lazy layer initialization, we need to move them to the correct device,
        # specify data types, and call them once to initialize the layers.
        G_init_input = torch.ones((1, params['init_noise_size'])).to(device)
        # D_init_input = torch.ones((1, params['time_series_length'], len(params['variables']))).to(device)
        D_init_input = torch.ones((1, params['time_series_length'], len(params['variables']))).to(device)
        G(G_init_input)
        D(D_init_input)

        # We'll use component-specific learning rates
        lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5])
        beta1 = trial.suggest_categorical("beta1", [0.0, 0.5, 0.9])
        if beta1 == 0.0:
            beta2 = 0.99
        elif beta1 == 0.5:
            beta2 = 0.9
        else:
            beta2 = 0.999
        betas = (beta1, beta2)
        optimizer_G = Adam(G.parameters(), lr=lr, betas=betas)
        optimizer_D = Adam(D.parameters(), lr=lr, betas=betas)

        plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'], transformer=transformer)
        trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                                critic_iterations=params['critic_iterations'],
                                plotter=plotter,
                                device=device,
                                silent=silent,
                                swa=False)

        plot_every  = max(1, params['epochs'] // 100)
        print_every = max(1, params['epochs'] // 30)
        trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=plot_every, print_every=print_every)

        # Save the trained models, parameters, and visualizations
        # Create a unique identifier string so we can save all models and plots with reasonable file
        # names. They don't need to be human readable as long as we save the params dictionary with
        # the model results so we can find the model directory given a set of tunable parameters.
        dirname = dirname or f'saved_models/cnn_retune/cnn_gnf{params["gen_num_filters"]}_gnl{params["gen_num_layers"]}_dnf{params["dis_num_filters"]}_dnl{params["dis_num_layers"]}_ns{noise_size}/'
        os.makedirs(dirname, exist_ok=True)

        # Save training visualizations
        iso = params['ISO']
        varnames_abbrev = ''.join([v.lower()[0] for v in params['variables']])
        trainer.save_training_gif(os.path.join(dirname, f'training_sde_{iso}_{varnames_abbrev}.gif'))

        # Save models
        torch.save(G.state_dict(), os.path.join(dirname, f'sde_gen_{iso}_{varnames_abbrev}.pt'))
        torch.save(D.state_dict(), os.path.join(dirname, f'sde_dis_{iso}_{varnames_abbrev}.pt'))

        if trainer._swa:
            torch.save(trainer.G_swa.state_dict(), os.path.join(dirname, f'sde_gen_swa_{iso}_{varnames_abbrev}.pt'))
            torch.save(trainer.D_swa.state_dict(), os.path.join(dirname, f'sde_dis_swa_{iso}_{varnames_abbrev}.pt'))

        # Save parameters
        params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # reuse the params_file name if it was specified, otherwise use the default naming scheme
        filename = os.path.join(dirname, f'params_sde_{iso}_{varnames_abbrev}.json')
        with open(filename, 'w') as f:
            json.dump(params, f)

        plot_model_results(G=G,
                           transformer=transformer,
                           model_type="CNN",
                           included_models="CNN",
                           varnames=params["variables"],
                           G_swa=trainer.G_swa if trainer._swa else None,
                           dirname=dirname)

        # Calculate the wasserstein distance between the real and generated data.
        wd = []
        real_data = transformer.inverse_transform(dataloader.dataset).detach().cpu().numpy()
        synth_data = G.sample(real_data.shape[0])
        synth_data = transformer.inverse_transform(synth_data).detach().cpu().numpy()
        for i in range(data_size):
            wd.append(wasserstein_distance(real_data[..., i].flatten(), synth_data[..., i].flatten()))

        return wd

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("cnn_gan_retune.log"))
    # study = optuna.create_study(directions=["minimize", "minimize", "minimize"], study_name=f"cnn_{noise_size}", storage=storage)
    # study.optimize(objective, n_trials=n_trials)

    study = optuna.load_study(study_name=f"cnn_{noise_size}", storage=storage)

    # Search the trials for the best for each objective
    best_totalload = None
    best_wind = None
    best_solar = None

    for trial in study.trials:
        values = trial.values
        if best_totalload is None or values[0] < best_totalload.values[0]:
            best_totalload = trial
        if best_wind is None or values[1] < best_wind.values[1]:
            best_wind = trial
        if best_solar is None or values[2] < best_solar.values[2]:
            best_solar = trial

    # fixed = optuna.trial.FixedTrial(best_totalload.params)
    # objective(fixed, dirname="saved_models/cnn_final/best_totalload")

    params = best_wind.params
    params["lr"] = 1e-4
    fixed = optuna.trial.FixedTrial(best_wind.params)
    objective(fixed, dirname="saved_models/cnn_final/best_wind_lr1e-4")
    params["lr"] = 1e-5
    fixed = optuna.trial.FixedTrial(best_wind.params)
    objective(fixed, dirname="saved_models/cnn_final/best_wind_lr1e-5")

    # fixed = optuna.trial.FixedTrial(best_solar.params)
    # objective(fixed, dirname="saved_models/cnn_final/best_solar")


if __name__ == '__main__':
    fire.Fire(tune_cnn_gan)
