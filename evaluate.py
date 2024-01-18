import numpy as np
import torch
import json
from dataloaders import get_dataloader
from scoring import distributions, hourly_distributions, autocorrelation, cross_correlation


def evaluate_cnn(params_file: str, **kwargs) -> None:
    # Load models inside evaluate_cnn to avoid conflicts with other model types
    from models.conv import Generator, Discriminator

    # with open('saved_models/cnn/cnn_params_ERCOT_sw.json', 'r') as f:
    with open(params_file, 'r') as f:
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

    # We'll use the CPU for evaluation. No real benefit to using the GPU here.
    G.cpu()
    D.cpu()
    G_init_input = G_init_input.cpu()
    D_init_input = D_init_input.cpu()

    # Initialize the lazy layers. Must be done before trying to load a saved model!
    G(G_init_input)
    D(D_init_input)

    # load the models
    G.load_state_dict(torch.load(f'saved_models/cnn/cnn_gen_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))
    D.load_state_dict(torch.load(f'saved_models/cnn/cnn_dis_{params["ISO"]}_{"".join([v.lower()[0] for v in params["variables"]])}.pt'))

    # Get historical data
    historical = pipeline.inverse_transform(dataloader.dataset.data.T).T
    historical = np.array(np.split(historical, historical.shape[1] // params['time_series_length'], axis=1))

    # Evaluate the model for basic statistical properties
    # First, we need to sample the generator. We'll use 1000 samples.
    latent_samples = G.sample_latent(len(historical))
    synthetic = G.transformed_sample(latent_samples)

    if kwargs.get('D', False):
        distributions(synthetic, historical, varnames=params['variables'])
    if kwargs.get('H', False):
        hourly_distributions(synthetic, historical, varnames=params['variables'])
    if kwargs.get('A', False):
        autocorrelation(synthetic, historical, varnames=params['variables'], lags=params['time_series_length'])
    if kwargs.get('X', False):
        cross_correlation(synthetic, historical, varnames=params['variables'], lags=params['time_series_length'])


if __name__ == '__main__':
    # 'saved_models/cnn/cnn_params_ERCOT_sw.json'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', action='store_true', default=False, help='evalaute distributions')
    parser.add_argument('-H', action='store_true', default=False, help='evalaute hourly distributions')
    parser.add_argument('-A', action='store_true', default=False, help='evalaute autocorrelation function')
    parser.add_argument('-X', action='store_true', default=False, help='evalaute cross-correlation function')
    parser.add_argument('--params-file', type=str, required=True, help='path to params file')
    args = vars(parser.parse_args())

    params_file = args.pop('params_file')  # removes params_file from args, leaving only the evaluation flags

    if 'cnn' in params_file:
        evaluate_cnn(params_file, **args)
    else:
        raise ValueError('Only CNN models are supported at this time.')
