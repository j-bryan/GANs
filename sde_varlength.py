import os
import json
from datetime import datetime
import fire

import numpy as np
import torch
from torch.optim import Adam, AdamW, RMSprop

from models.sde import Generator, Discriminator, SdeGeneratorConfig, CdeDiscriminatorConfig
from training import WGANGPTrainer, WGANLPTrainer
from dataloaders import get_sde_dataloader_varlength
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device

from dataclasses import dataclass
from models.layers import FFNNConfig

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


def tune_sdegan(n_trials: int = 100,
                epochs: int = 2000,
                batch_size: int = 256,
                device: str = "cpu",
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
    dataloader, transformer = get_sde_dataloader_varlength(
        iso="ERCOT",
        varname=["TOTALLOAD", "WIND", "SOLAR"],
        segment_size=segment_size,
        max_segments_per_sample=7,
        batch_size=batch_size,
        device=device
    )
    critic_iterations = max(1, min(10, len(dataloader)))

    def objective(trial: optuna.Trial):
        params = {
            "ISO":                "ERCOT",
            "variables":          ["TOTALLOAD", "WIND", "SOLAR"],
            "time_features":      ["HOD"],
            "time_series_length": segment_size,
            "critic_iterations":  critic_iterations,
            "penalty_weight":     10.0,
            "epochs":             epochs,
            "random_seed":        12345,
            "batch_size":         batch_size
        }

        readout_activations = {
            "TOTALLOAD": "identity",    # output in (-inf, inf)
            "WIND":      "sigmoid",     # output in (0, 1)
            "SOLAR":     "hardsigmoid"  # output in [0, 1]
        }

        data_size = len(params["variables"])
        time_size = len(params["time_features"])

        sde_hidden_size = trial.suggest_int("sde_hidden_size", data_size, 128)
        initial_noise_size = sde_hidden_size
        sde_noise_size = sde_hidden_size  # must be same as hidden size for diagonal noise
        sde_noise_type = "diagonal"

        # For diagonal noise, we require the SDE noise size and the SDE hidden size to be the same.
        # If they're not, we'll prune the trial.
        # TODO: check to make sure we actually get some "diagonal" trials
        if sde_noise_type == "diagonal" and sde_noise_size != sde_hidden_size:
            raise optuna.TrialPruned

        num_units = trial.suggest_categorical("num_units", [64, 128, 256, 512])
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)

        gen_noise_embed_config = FFNNConfig(
            in_size=initial_noise_size,
            num_hidden_layers=2,
            num_units=sde_hidden_size,
            out_size=sde_hidden_size
        )
        gen_drift_config = FFNNConfig(
            in_size=sde_hidden_size + time_size,
            num_hidden_layers=num_hidden_layers,
            num_units=num_units,
            out_size=sde_hidden_size,
            final_activation="tanh"
        )
        diffusion_out_size = sde_hidden_size if sde_noise_type == "diagonal" else sde_hidden_size * sde_noise_size
        gen_diffusion_config = FFNNConfig(
            in_size=sde_hidden_size + time_size,
            num_hidden_layers=num_hidden_layers,
            num_units=num_units,
            out_size=diffusion_out_size,
            final_activation="tanh"
        )
        gen_readout_config = FFNNConfig(
            in_size=sde_hidden_size + time_size,
            num_hidden_layers=trial.suggest_int("readout_num_hidden_layers", 0, 3),
            num_units=trial.suggest_categorical("readout_num_units", [16, 32, 64, 128, 256, 512]),
            out_size=data_size,
            final_activation=[readout_activations[v] for v in params["variables"]]
        )
        sde_generator_config = SdeGeneratorConfig(
            noise_type=sde_noise_type,
            sde_type="stratonovich",
            time_steps=params["time_series_length"],
            time_size=time_size,
            data_size=data_size,
            init_noise_size=initial_noise_size,
            noise_size=sde_noise_size,
            hidden_size=sde_hidden_size,
            drift_config=gen_drift_config,
            diffusion_config=gen_diffusion_config,
            embed_config=gen_noise_embed_config,
            readout_config=gen_readout_config,
            return_ty=True
        )
        discriminator_config = CdeDiscriminatorConfig(
            data_size=data_size,
            hidden_size=sde_hidden_size,
            mlp_size=num_units,
            num_layers=num_hidden_layers
        )
        gen_opt_config = AdamConfig(
            lr=5e-5,
            betas=(0.5, 0.9),
            weight_decay=1e-2
        )
        dis_opt_config = AdamConfig(
            lr=1e-4,
            betas=(0.5, 0.9),
            weight_decay=1e-2
        )

        params.update(sde_generator_config.to_dict())
        params.update(discriminator_config.to_dict(prefix="dis"))
        params.update(gen_opt_config.to_dict(prefix="gen"))
        params.update(dis_opt_config.to_dict(prefix="dis"))

        if isinstance(params['variables'], str):
            params['variables'] = [params['variables']]
        # seed for reproducibility
        np.random.seed(params['random_seed'])
        torch.manual_seed(params['random_seed'])

        G = Generator(sde_generator_config).to(device)
        # D = DiscriminatorSimple(discriminator_config).to(device)
        # D = Discriminator(num_filters=params["dis_num_filters"], num_layers=params["dis_num_layers"]).to(device)
        # Initialize the lazy layers
        # D(torch.randn(1, params["time_series_length"], data_size).to(device))
        D = Discriminator(discriminator_config).to(device)

        # We'll use component-specific learning rates
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
        if optimizer == "adam":
            optimizer_G = Adam(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
            optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])
        elif optimizer == "adamw":
            optimizer_G = AdamW(G.parameters(), lr=params['gen_lr'], betas=params['gen_betas'])
            optimizer_D = AdamW(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])
        elif optimizer == "rmsprop":
            optimizer_G = RMSprop(G.parameters(), lr=params['gen_lr'], weight_decay=params["gen_weight_decay"])
            optimizer_D = RMSprop(D.parameters(), lr=params['dis_lr'], weight_decay=params["dis_weight_decay"])

        plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'], transformer=transformer)
        # trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
        #                         critic_iterations=params['critic_iterations'],
        #                         plotter=plotter,
        #                         device=device,
        #                         silent=silent,
        #                         swa=False)
        trainer = WGANLPTrainer(G, D, optimizer_G, optimizer_D,
                                penalty_weight=params['penalty_weight'],
                                critic_iterations=params['critic_iterations'],
                                generator_iterations=1,
                                plotter=plotter,
                                device=device)

        plot_every  = max(1, params['epochs'] // 100)
        print_every = max(1, params['epochs'] // 30)

        # Before we train the model, check to see if the parameters have already been evaluated.
        # If they have, we can skip training and return the existing value.
        if not isinstance(trial, optuna.trial.FixedTrial):
            states_to_consider = (optuna.trial.TrialState.COMPLETE,)
            trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
            # Check whether we already evaluated the sampled `(x, y)`.
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    # Use the existing value as trial duplicated the parameters.
                    return t.value

        trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=plot_every, print_every=print_every)

        # Save the trained models, parameters, and visualizations
        # Create a unique identifier string so we can save all models and plots with reasonable file
        # names. They don't need to be human readable as long as we save the params dictionary with
        # the model results so we can find the model directory given a set of tunable parameters.
        dirname = f'saved_models/sde_varlen_hidden{sde_hidden_size}_layers{num_hidden_layers}_units{num_units}_opt{optimizer.title()}/'
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Save training visualizations
        iso = params['ISO']
        varnames_abbrev = ''.join([v.lower()[0] for v in params['variables']])
        trainer.save_training_gif(dirname + f'training_sde_{iso}_{varnames_abbrev}.gif')

        # Save models
        torch.save(G.state_dict(), dirname + f'sde_gen_{iso}_{varnames_abbrev}.pt')
        torch.save(D.state_dict(), dirname + f'sde_dis_{iso}_{varnames_abbrev}.pt')

        if trainer._swa:
            torch.save(trainer.G_swa.state_dict(), dirname + f'sde_gen_swa_{iso}_{varnames_abbrev}.pt')
            torch.save(trainer.D_swa.state_dict(), dirname + f'sde_dis_swa_{iso}_{varnames_abbrev}.pt')

        # Save parameters
        params['model_save_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # reuse the params_file name if it was specified, otherwise use the default naming scheme
        filename = dirname + f'params_sde_{iso}_{varnames_abbrev}.json'
        with open(filename, 'w') as f:
            json.dump(params, f)

        plot_model_results(G, transformer, params["variables"], dirname=dirname)

        return 0

    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("sde_gan.log"))
    # study = optuna.create_study(direction="minimize", study_name="sde_gan", storage=storage, load_if_exists=True)
    # study.optimize(objective, n_trials=n_trials)
    fixed = optuna.trial.FixedTrial({
        "sde_hidden_size": 16,
        "num_units": 64,
        "num_hidden_layers": 2,
        "readout_num_hidden_layers": 2,
        "readout_num_units": 64,
        "optimizer": "adam"
    })
    objective(fixed)


if __name__ == '__main__':
    fire.Fire(tune_sdegan)
