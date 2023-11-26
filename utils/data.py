import torch
from torch import FloatTensor
from torch.nn import Module
import numpy as np
import torch.distributions as td
from copy import deepcopy
from typing import Union, List, Tuple
from utils.replays import ReplayBuffer
from utils.scalers import StandardScaler


def preprocess_sac_batch_oto(offline_buffer: ReplayBuffer, model_buffer: ReplayBuffer, online_buffer: ReplayBuffer,
                             batch_size: int, real_ratio: float, online_ratio: float) -> List[FloatTensor]:
    # TODO: fix the below to be dependent on batch_size kwarg
    offline_bs = int(batch_size * real_ratio)
    model_bs = int(batch_size * (1 - real_ratio))
    online_bs = int(batch_size * online_ratio)

    offline_batch = offline_buffer.sample(offline_bs, rl=True)
    model_batch = model_buffer.sample(model_bs, rl=True)
    online_batch = online_buffer.sample(online_bs, rl=True)

    batch = [
        torch.cat((offline_item, model_item, online_item), dim=0)
        for offline_item, model_item, online_item in zip(offline_batch, model_batch, online_batch)
    ]

    return batch


def preprocess_sac_batch(env_replay_buffer: ReplayBuffer, model_replay_buffer: ReplayBuffer, batch_size: int,
                         real_ratio: float) -> List[FloatTensor]:
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size, rl=True)
    model_batch = model_replay_buffer.sample(model_batch_size, rl=True)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch


def preprocess_sac_batch_latent(env_replay_buffer: ReplayBuffer, model_replay_buffer: ReplayBuffer, batch_size: int,
                                real_ratio: float, scaler: StandardScaler, state_encoder: Module) -> List[FloatTensor]:
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size, rl=True)
    model_batch = model_replay_buffer.sample(model_batch_size, rl=True)

    # States in the model-replay buffer are already in the transformed latent space. However,
    # we need to perform the transformation to the real states in the env_replay_buffer
    s, a, ns, r, nd = env_batch
    s = scaler.transform(torch.cat([s, torch.ones_like(a)], dim=-1))[:, :s.shape[-1]]
    ns = scaler.transform(torch.cat([ns, torch.ones_like(a)], dim=-1))[:, :ns.shape[-1]]

    with torch.no_grad():
        s = state_encoder(s)
        ns = state_encoder(ns)

    env_batch = (s, a, ns, r, nd)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch


def symlog(x: FloatTensor) -> FloatTensor:
    return torch.sign(x) * torch.log(x.abs() + 1)


def inv_symlog(x: FloatTensor) -> FloatTensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)
