import torch
from torch import nn, FloatTensor
import torch.nn.functional as F
import torch.distributions as td
import math
from typing import Union, List
import jax
from jax import numpy as jnp
from flax import linen as fnn
import distrax
# from rlax._src.distributions import squashed_gaussian


class DistLayer(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, dist: str) -> None:
        super().__init__()
        self.dist = dist
        self.lin_proj = nn.Linear(input_shape, output_shape)

        if dist in ['normal', 'trunc_normal']:
            self.std_proj = nn.Linear(input_shape, output_shape)

            self.min_logvar = nn.Parameter(
                torch.ones(output_shape) * -10,
                requires_grad=True
            )
            self.max_logvar = nn.Parameter(
                torch.ones(output_shape) * 0.5,
                requires_grad=True
            )

    def forward(
            self, x: FloatTensor, moments: bool
    ) -> Union[List[FloatTensor], td.Distribution, Union[FloatTensor, int]]:
        mu = self.lin_proj(x)

        if self.dist == 'normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar

            else:
                dist = td.Normal(mu, torch.sqrt(torch.exp(logvar)))
                return dist

        elif self.dist == 'mse':
            if moments:
                return mu, 1

            else:
                dist = td.Normal(mu, 1.0)
                return dist

        elif self.dist == 'normal_var_adjust':
            std = self.std_proj(x)
            std = 2 * torch.sigmoid((std + self.init_std) / 2)
            if moments:
                pass

        elif self.dist == 'trunc_normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar

            else:
                dist = SquashedNormal(mu, torch.sqrt(torch.exp(logvar)))
                return dist


class TanhTransform(td.transforms.Transform):
    domain = td.constraints.real
    codomain = td.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1) -> None:
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x: FloatTensor) -> FloatTensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TanhTransform)

    def _call(self, x: FloatTensor) -> FloatTensor:
        return x.tanh()

    def _inverse(self, y: FloatTensor) -> FloatTensor:
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x: FloatTensor, y: FloatTensor) -> FloatTensor:
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7  # noqa
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(td.transformed_distribution.TransformedDistribution):
    def __init__(self, loc: FloatTensor, scale: FloatTensor) -> None:
        self.loc = loc
        self.scale = scale

        self.base_dist = td.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self) -> FloatTensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self) -> FloatTensor:
        return self.base_dist.entropy()


class DistLayerJAX(fnn.Module):
    input_shape: int
    output_shape: int
    dist: str

    def setup(self) -> None:
        self.lin_proj = fnn.Dense(self.output_shape)

        if self.dist in ['normal', 'trunc_normal']:
            self.std_proj = fnn.Dense(self.output_shape)

            self.min_logvar = self.param('min_logvar', fnn.initializers.constant(-10), self.output_shape)
            self.max_logvar = self.param('max_logvar', fnn.initializers.constant(0.5), self.output_shape)

    def __call__(self, x: FloatTensor, moments: bool):
        mu = self.lin_proj(x)

        if self.dist == 'normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - fnn.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + fnn.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar
            else:
                dist = distrax.Normal(mu, jnp.sqrt(jnp.exp(logvar)))
                return dist

        elif self.dist == 'mse':
            if moments:
                return mu, 1
            else:
                dist = distrax.Normal(mu, 1.0)
                return dist

        elif self.dist == 'trunc_normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - fnn.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + fnn.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar
            else:
                dist = SquashedNormalJAX(mu, jnp.sqrt(jnp.exp(logvar)))
                return dist


class SquashedNormalJAX(distrax.Transformed):
    """Code taken from: https://github.com/Howuhh/sac-n-jax/blob/main/sac_n_jax_flax.py#L91"""
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

# distrax.
# test = DistLayerJAX(32, 2, 'trunc_normal')
# print(test)
#
# x = jax.random.normal(jax.random.PRNGKey(42), (10, 32))
# print(x.shape)
# variables = test.init(jax.random.key(0), x, moments=False)
# # print(f'VARS: {variables}')
# # print('VARS:')
# # for k, v in variables['params'].items():
# #     print(k)
# mu, logvar = test.apply(variables, x, moments=True)
# print(mu)
# print()
# print(logvar)
#
# dist = test.apply(variables, x, moments=False)
# print(dist)
# print(dist.sample(seed=42))
