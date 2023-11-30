from typing import Union, List, Tuple
import numpy as np
from torch import FloatTensor


class StandardScaler:
    """ Used to calculate mean, std and normalize data. """

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data: Union[FloatTensor, np.array], traj: bool = False) -> None:
        """ Calculate mean and std for given data."""
        if not traj:
            self.mean = data.mean(0, keepdim=True)  # calculate mean among batch
            self.std = data.std(0, keepdim=True)
            self.std[self.std < 1e-12] = 1.0
        else:
            self.mean = data.mean([0, 1], keepdim=True)  # calculate mean among batch
            self.std = data.std([0, 1], keepdim=True)
            self.std[self.std < 1e-12] = 1.0

    def transform(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        """ Normalization. """
        return (data - self.mean) / self.std

    def transform_std_only(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        return data / self.std

    def inverse_transform(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        return data * self.std + self.mean


class StandardScalerJAX:
    """ Used to calculate mean, std and normalize data. """

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data: Union[FloatTensor, np.array], traj: bool = False) -> None:
        """ Calculate mean and std for given data."""
        if not traj:
            self.mean = data.mean(0, keepdims=True)  # calculate mean among batch
            self.std = data.std(0, keepdims=True)
            self.std = self.std.at[self.std < 1e-12].set(1.0)
        else:
            self.mean = data.mean([0, 1], keepdims=True)  # calculate mean among batch
            self.std = data.std([0, 1], keepdims=True)
            self.std = self.std.at[self.std < 1e-12].set(1.0)

    def transform(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        """ Normalization. """
        return (data - self.mean) / self.std

    def transform_std_only(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        return data / self.std

    def inverse_transform(self, data: Union[FloatTensor, np.array]) -> Union[FloatTensor, np.array]:
        return data * self.std + self.mean
