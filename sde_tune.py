import os
import json
from datetime import datetime
import uuid
import fire

import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.optim import Adam, AdamW, RMSprop, Adadelta

from models.sde import Generator, DiscriminatorSimple, SdeGeneratorConfig
from training import WGANGPTrainer
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


def tune_sdegan(n_trials: int = 64,
                epochs: int = 10000,
                batch_size: int = 1826,
                device: str = "cuda",
                storage: str = "sde_retune_fine.log",
                study_name: str = "sde_gan",
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
    critic_iterations = 5

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

        # sde_hidden_size = trial.suggest_categorical("sde_hidden_size", [16, 32, 64, 128, 256])
        # sde_hidden_size = trial.suggest_categorical("sde_hidden_size", [16, 32])
        sde_hidden_size = 32
        initial_noise_size = sde_hidden_size
        sde_noise_size = sde_hidden_size  # must be same as hidden size for diagonal noise
        sde_noise_type = "diagonal"
        # time_dependent_readout = trial.suggest_categorical("time_dependent_readout", [True, False])
        time_dependent_readout = False
        time_dependent_drift = False
        time_dependent_diffusion = False

        # For diagonal noise, we require the SDE noise size and the SDE hidden size to be the same.
        # If they're not, we'll prune the trial.
        # TODO: check to make sure we actually get some "diagonal" trials
        if sde_noise_type == "diagonal" and sde_noise_size != sde_hidden_size:
            raise optuna.TrialPruned

        # num_units = trial.suggest_categorical("num_units", [64, 128, 256, 512])
        # num_units = trial.suggest_categorical("num_units", [64, 128])
        num_units = 128
        # num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 4)
        num_hidden_layers = 3

        gen_noise_embed_config = FFNNConfig(
            in_size=initial_noise_size,
            num_hidden_layers=2,
            num_units=sde_hidden_size,
            out_size=sde_hidden_size
        )
        drift_in_size = sde_hidden_size + time_size if time_dependent_drift else sde_hidden_size
        gen_drift_config = FFNNConfig(
            in_size=drift_in_size,
            num_hidden_layers=num_hidden_layers,
            num_units=num_units,
            out_size=sde_hidden_size,
            final_activation="tanh"
        )
        diffusion_in_size = sde_hidden_size + time_size if time_dependent_diffusion else sde_hidden_size
        diffusion_out_size = sde_hidden_size if sde_noise_type == "diagonal" else sde_hidden_size * sde_noise_size
        gen_diffusion_config = FFNNConfig(
            in_size=diffusion_in_size,
            num_hidden_layers=num_hidden_layers,
            num_units=num_units,
            out_size=diffusion_out_size,
            final_activation="tanh"
        )
        readout_in_size = sde_hidden_size + time_size if time_dependent_readout else sde_hidden_size
        gen_readout_config = FFNNConfig(
            in_size=readout_in_size,
            # num_hidden_layers=trial.suggest_int("readout_num_hidden_layers", 0, 1),
            num_hidden_layers=0,
            # num_units=trial.suggest_categorical("readout_num_units", [128, 256, 512]),
            num_units=1,  # doesn't get used since num_hidden_layers is 0
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
            time_dependent_readout=time_dependent_readout,
            time_dependent_drift=time_dependent_drift,
            time_dependent_diffusion=time_dependent_diffusion
        )
        discriminator_config = FFNNConfig(
            in_size=data_size * params["time_series_length"],
            num_hidden_layers=trial.suggest_int("dis_num_hidden_layers", 2, 3),
            num_units=trial.suggest_categorical("dis_num_units", [128, 256, 512, 1024]),
            out_size=1
        )
        gen_lr = trial.suggest_categorical("genopt_lr", [1e-5, 5e-5, 1e-4])
        params["genopt_init_lr"] = gen_lr * trial.suggest_categorical("genopt_lr_init_mult", [1.0, 2.0, 5.0, 10.0])
        dis_lr_mult = trial.suggest_categorical("disopt_lr_mult", [1.0, 2.0, 5.0, 10.0])
        params["genopt_lr"] = gen_lr
        params["disopt_lr"] = gen_lr * dis_lr_mult
        # gen_opt_config = AdamConfig(
        #     lr=gen_lr,
        #     betas=(0.0, 0.99),
        #     weight_decay=0.0
        # )
        # dis_opt_config = AdamConfig(
        #     lr=gen_lr*dis_lr_mult,
        #     betas=(0.0, 0.99),
        #     weight_decay=0.0
        # )

        params.update(sde_generator_config.to_dict())
        params.update(discriminator_config.to_dict(prefix="dis"))
        # params.update(gen_opt_config.to_dict(prefix="genopt"))
        # params.update(dis_opt_config.to_dict(prefix="disopt"))

        if isinstance(params['variables'], str):
            params['variables'] = [params['variables']]
        # seed for reproducibility
        np.random.seed(params['random_seed'])
        torch.manual_seed(params['random_seed'])

        G = Generator(sde_generator_config).to(device)
        D = DiscriminatorSimple(discriminator_config).to(device)

        # Trying out the Adadelta optimizer based on the suggestions in the torchsde SDE-GAN example
        # gen_weight_decay = trial.suggest_categorical("genopt_weight_decay", [0.0, 1e-4, 1e-3, 1e-2])
        # dis_weight_decay = trial.suggest_categorical("disopt_weight_decay", [0.0, 1e-4, 1e-3, 1e-2])
        # params["genopt_weight_decay"] = gen_weight_decay
        # params["disopt_weight_decay"] = dis_weight_decay
        # optimizer_G = Adadelta([
        #     {"params": G._initial.parameters(), "lr": params["genopt_init_lr"]},
        #     {"params": G._func.parameters()},
        #     {"params": G._readout.parameters()}
        # ], lr=params["genopt_lr"], weight_decay=gen_weight_decay)
        # optimizer_D = Adadelta(D.parameters(), lr=params['disopt_lr'], weight_decay=dis_weight_decay)
        # NOTE: Contrary to the torchsde example, I didn't have much luck with Adadelta. Adam has
        # been working the best for me so far.
        genopt_beta1 = trial.suggest_categorical("genopt_beta1", [0.0, 0.5, 0.9])
        if genopt_beta1 == 0.0:
            genopt_betas = (genopt_beta1, 0.99)
        elif genopt_beta1 == 0.5:
            genopt_betas = (genopt_beta1, 0.9)
        else:
            genopt_betas = (genopt_beta1, 0.999)
        disopt_betas = genopt_betas
        params["genopt_betas"] = genopt_betas
        params["disopt_betas"] = disopt_betas
        optimizer_G = Adam([
            {"params": G._initial.parameters(), "lr": params["genopt_init_lr"]},
            {"params": G._func.parameters()},
            {"params": G._readout.parameters()}
        ], lr=params["genopt_lr"], betas=params["genopt_betas"])
        optimizer_D = Adam(D.parameters(), lr=params['disopt_lr'], betas=params["disopt_betas"])

        plotter = SDETrainingPlotter(['G', 'D'], varnames=params['variables'], transformer=transformer)
        trainer = WGANGPTrainer(G, D, optimizer_G, optimizer_D,
                                critic_iterations=params['critic_iterations'],
                                plotter=plotter,
                                device=device,
                                silent=silent,
                                swa=True)

        plot_every  = max(1, params['epochs'] // 100)
        print_every = max(1, params['epochs'] // 30)

        # Before we train the model, check to see if the parameters have already been evaluated.
        # If they have, we can skip training and return the existing value.
        # I'm not sure what errors might get thrown here, but I'd rather catch them than have it
        # crash the whole optimization. Better to just rerun the point in that case.
        try:
            if not isinstance(trial, optuna.trial.FixedTrial):
                states_to_consider = (optuna.trial.TrialState.COMPLETE,)
                trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
                # Check whether we already evaluated the sampled `(x, y)`.
                for t in reversed(trials_to_consider):
                    if trial.params == t.params:
                        # Use the existing value as trial duplicated the parameters.
                        return t.value
        except:
            pass
        print(trial.params)

        # import json
        # with open("saved_models/sde_hiddensize16_numunits64_numlayers3/params_sde_ERCOT_tws.json", "r") as f:
        #     saved_params = json.load(f)
        # all_keys = list(set(params.keys()) | set(saved_params.keys()))
        # for k in all_keys:
        #     if k not in params:
        #         print(k, "not in params...", saved_params[k])
        #         continue
        #     if k not in saved_params:
        #         print(k, "not in saved_params...", params[k])
        #         continue
        #     print(f"{k}\t{params[k]}\t{saved_params[k]}")

        # exit()

        trainer.train(data_loader=dataloader, epochs=params['epochs'], plot_every=plot_every, print_every=print_every)

        # Save the trained models, parameters, and visualizations
        # Create a unique identifier string so we can save all models and plots with reasonable file
        # names. They don't need to be human readable as long as we save the params dictionary with
        # the model results so we can find the model directory given a set of tunable parameters.
        # dirname = f'saved_models/sde_{sde_hidden_size}_{num_hidden_layers}_{num_units}_opt{optimizer.title()}/'
        param_str = ""
        for k, v in trial.params.items():
            # take the first letter of underscore delimited words in the key
            shortened_key = ''.join([k[0].lower() for k in k.split('_')])
            param_str += f"{shortened_key}{v}"
        dirname = f"saved_models/dynamical/"
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

        # Calculate Wasserstein distance between the real and generated data for each variable
        # and return the average. Use the score of the SWA model if it was used.
        data = dataloader.dataset.data
        num_samples = data.size(0)

        if trainer._swa:
            latent_samples = trainer.G.sample_latent(num_samples)
            G = trainer.G_swa
        else:
            G = trainer.G

        samples = G(latent_samples).detach()
        data = transformer.inverse_transform(data).cpu().numpy()
        samples = transformer.inverse_transform(samples).cpu().numpy()
        wd = [wasserstein_distance(data[..., i].ravel(), samples[..., i].ravel()) for i in range(data_size)]

        return wd

    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage))
    # study = optuna.create_study(directions=["minimize", "minimize", "minimize"], study_name=study_name, storage=storage, load_if_exists=True)
    # study.optimize(objective, n_trials=n_trials)
    # with open("../best_sde_model/params_sde_ERCOT_tws.json", "r") as f:
    #     saved_params = json.load(f)

    # best_params = {
    #     "sde_hidden_size": saved_params["hidden_size"],
    #     "num_units": saved_params["drift_num_units"],
    #     "num_hidden_layers": saved_params["drift_num_hidden_layers"],
    #     "dis_num_layers": saved_params["dis_num_layers"],
    #     "dis_num_units": saved_params["dis_num_filters"],
    #     "readout_num_hidden_layers": saved_params["readout_num_hidden_layers"],
    #     "readout_num_units": saved_params["readout_num_units"]
    # }

    best_params = {'dis_num_hidden_layers': 3, 'dis_num_units': 512, 'genopt_lr': 5e-05, 'genopt_lr_init_mult': 10.0, 'disopt_lr_mult': 5.0, 'genopt_beta1': 0.0}
    fixed_trial = optuna.trial.FixedTrial(best_params)
    objective(fixed_trial)


if __name__ == '__main__':
    fire.Fire(tune_sdegan)
