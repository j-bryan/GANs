import numpy as np
import torch


def count_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_accelerator_device() -> str:
    """
    Checks for CUDA (NVIDIA GPUs) and MPS (Apple silicon integrated GPU)
    availability before defaulting back to CPU if neither accelerator is present.

    Returns
    -------
    device : str
        The device to use for pytorch objects.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        print('WARNING: No GPU detected. Falling back to CPU. This might be slow!')
        device = 'cpu'
    return device
