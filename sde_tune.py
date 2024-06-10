import os
import json
from datetime import datetime
import fire
from uuid import uuid4

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
    dataloader, test_dataloader, _, transformer = get_sde_dataloader(
        iso="ERCOT",
        varname=["TOTALLOAD", "WIND", "SOLAR"],
        segment_size=segment_size,
        batch_size=batch_size,
        device=device,
        test_size=0.2,
        valid_size=0.1,
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

        hidden_size = trial.suggest_int("hidden_size", data_size, 128)
        initial_noise_size = hidden_size
        sde_noise_size = trial.suggest_int("sde_noise_size", data_size, 128)
        sde_noise_type = "general"

        # For diagonal noise, we require the SDE noise size and the SDE hidden size to be the same.
        # If they're not, we'll prune the trial.
        # TODO: check to make sure we actually get some "diagonal" trials
        if sde_noise_type == "diagonal" and sde_noise_size != hidden_size:
            raise optuna.TrialPruned

        gen_noise_embed_config = FFNNConfig(
            in_size=initial_noise_size,
            num_layers=trial.suggest_int("gen_noise_embed_num_layers", 1, 3),
            num_units=trial.suggest_int("gen_noise_embed_num_units", 16, 128),
            out_size=hidden_size
        )
        gen_drift_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=trial.suggest_int("gen_drift_num_layers", 1, 3),
            num_units=trial.suggest_int("gen_drift_num_units", 16, 128),
            out_size=hidden_size,
            final_activation="tanh"
        )
        diffusion_out_size = hidden_size if sde_noise_type == "diagonal" else hidden_size * sde_noise_size
        gen_diffusion_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=trial.suggest_int("gen_diffusion_num_layers", 1, 3),
            num_units=trial.suggest_int("gen_diffusion_num_units", 16, 128),
            out_size=diffusion_out_size,
            final_activation="tanh"
        )
        gen_readout_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=trial.suggest_int("gen_readout_num_layers", 1, 3),
            num_units=trial.suggest_int("gen_readout_num_units", 16, 128),
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
            hidden_size=hidden_size,
            drift_config=gen_drift_config,
            diffusion_config=gen_diffusion_config,
            embed_config=gen_noise_embed_config,
            readout_config=gen_readout_config
        )
        discriminator_config = FFNNConfig(
            in_size=data_size * params["time_series_length"],
            num_layers=trial.suggest_int("discriminator_num_layers", 1, 5),
            num_units=trial.suggest_int("discriminator_num_units", 16, 256),
            out_size=1
        )
        gen_opt_config = AdamConfig(
            lr=trial.suggest_float("gen_opt_sde_lr", 5e-6, 5e-4),
            betas=(
                trial.suggest_float("gen_opt_beta1", 0.5, 0.9),
                trial.suggest_float("gen_opt_beta2", 0.9, 0.999)),
            weight_decay=trial.suggest_float("gen_opt_weight_decay", 1e-6, 1e-3, log=True)
        )
        dis_opt_config = AdamConfig(
            lr=trial.suggest_float("dis_opt_lr", 5e-6, 5e-4),
            betas=(
                trial.suggest_float("dis_opt_beta1", 0.5, 0.9),
                trial.suggest_float("dis_opt_beta2", 0.9, 0.999)
            ),
            weight_decay=trial.suggest_float("dis_opt_weight_decay", 1e-6, 1e-3, log=True)
        )

        params.update(sde_generator_config.to_dict())
        params.update(discriminator_config.to_dict(prefix="dis"))
        params.update(gen_opt_config.to_dict(prefix="gen"))
        params.update(dis_opt_config.to_dict(prefix="dis"))

        params["gen_opt_noise_embed_lr"] = trial.suggest_float("gen_opt_noise_embed_lr", 5e-6, 5e-4)
        params["gen_opt_readout_lr"]     = trial.suggest_float("gen_opt_readout_lr",     5e-6, 5e-4)

        if isinstance(params['variables'], str):
            params['variables'] = [params['variables']]
        # seed for reproducibility
        np.random.seed(params['random_seed'])
        torch.manual_seed(params['random_seed'])

        G = Generator(sde_generator_config).to(device)
        D = DiscriminatorSimple(discriminator_config).to(device)

        # We'll use component-specific learning rates
        optimizer_G = Adam([
                {"params": G._initial.parameters(), "lr": params["gen_opt_noise_embed_lr"]},
                {"params": G._func.parameters(), "lr": params["gen_lr"]},
                {"params": G._readout.parameters(), "lr": params["gen_opt_readout_lr"]}
            ], lr=params['gen_lr'], betas=params['gen_betas'])
        optimizer_D = Adam(D.parameters(), lr=params['dis_lr'], betas=params['dis_betas'])

        plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'], transformer=transformer)
        trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                                critic_iterations=params['critic_iterations'],
                                plotter=plotter,
                                device=device,
                                silent=silent,
                                swa=trial.suggest_categorical("use_swa", [True, False]))

        plot_every  = max(1, params['epochs'] // 100)
        print_every = max(1, params['epochs'] // 30)
        trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=plot_every, print_every=print_every)

        # Save the trained models, parameters, and visualizations
        # Create a unique identifier string so we can save all models and plots with reasonable file
        # names. They don't need to be human readable as long as we save the params dictionary with
        # the model results so we can find the model directory given a set of tunable parameters.
        id = uuid4().hex
        dirname = f'saved_models/sde_{id}/'
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

        plot_model_results(G, transformer, params["variables"],
                        G_swa=trainer.G_swa if trainer._swa else None,
                        dirname=dirname)

        # Evaluate the model on the test set
        # We do this in a somewhat ad-hoc way by calculating Wasserstein distances of every metric
        # we can conceivably care about, scaled by the support of the distribution of that metric.
        # We calculate metrics for samples, first differences, autocorrelation, and cross-correlation
        # of every variable in the dataset.
        G_swa = trainer.G_swa if trainer._swa else None
        metrics = calculate_metrics(G, test_dataloader, transformer, params["variables"], G_swa)
        # Among the metrics returned here is the "total" metric, which is a sum of all of the weighted
        # Wasserstein distances of the other metrics. This is the metric we'll use to evaluate the model.

        return metrics["total"]

    storage = f"sqlite:///optuna.db"
    study = optuna.create_study(direction="minimize", study_name="sde_gan", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)


if __name__ == '__main__':
    fire.Fire(tune_sdegan)
