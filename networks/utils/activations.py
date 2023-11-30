from torch import nn
from typing import Union, Callable
from flax import linen as fnn


def get_activation(activation_name: str) -> Union[nn.Module, ValueError]:
    """

    Args:
        activation_name:

    Returns:

    """
    if activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f'{activation_name} not supported currently.')


def get_activation_jax(activation_name: str) -> Union[Callable, ValueError]:
    """"""
    if activation_name == 'tanh':
        return fnn.tanh
    elif activation_name == 'relu':
        return fnn.relu
    elif activation_name == 'elu':
        return fnn.elu
    else:
        raise ValueError(f'{activation_name} not supported currently.')
