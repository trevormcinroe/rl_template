from torch import nn
from typing import Union


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
