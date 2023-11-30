# from networks.forward_models import MLPJAX
import jax
from jax import numpy as jnp
from networks.ensembles import DynamicsEnsembleJAX, DynamicsEnsemble
import distrax
from utils.scalers import StandardScalerJAX, StandardScaler
import d4rl
import gym
from utils.replays import OfflineReplay, OfflineReplayJAX
from typing import NamedTuple, Dict, Callable
from flax import linen as fnn
import wandb
import json
import optax
from utils.data import TrainingState
import time


import argparse
args = argparse.ArgumentParser()
args.add_argument('--jax', action='store_true')
args = args.parse_args()

"""LOGGING"""
# with open('../wandb.txt', 'r') as f:
#     API_KEY = json.load(f)['api_key']
#
# import os
# os.environ['WANDB_API_KEY'] = API_KEY
# os.environ['WANDB_DIR'] = './wandb'
# os.environ['WANDB_CONFIG_DIR'] = './wandb'
#
# wandb.init(
#             project='jax-testing',
#             entity='trevor-mcinroe',
#             name=f'{"jax" if args.jax else "torch"}-hcrandom-mbpo-training',
#         )


env = gym.make('halfcheetah-random-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

env.reset()
start = time.time()

if args.jax:
    offline_replay = OfflineReplayJAX(env, 'cpu')
    obs, acts, next_obs, rewards, not_dones = offline_replay.sample(2, False)

    mbpo = DynamicsEnsembleJAX(7, obs_dim, act_dim, [200 for _ in range(4)], 'elu', False, 'normal',
                               50000, True, True, 512, 1e-3, 10, 5, None, False, None, None)

    ensemble_optim = optax.chain(optax.adamw(learning_rate=1e-3, weight_decay=1e-5))
    mbpo.scaler.fit(jnp.concatenate([obs, acts], -1))

    params = mbpo.init(jax.random.key(0), obs, acts, True)

    batch_size = 512
    validation_ratio = 0.2
    val_size = int(batch_size * validation_ratio)
    train_size = batch_size - val_size
    train_batch, val_batch = offline_replay.random_split(val_size, batch_size * 10)

    print(type(mbpo.variables))
    qqq

    opt_state = ensemble_optim.init(params)
    # print(opt_state)

    # TODO: the opt_state needs to be a tuple??
    training_state = TrainingState(params=params, opt_state=opt_state, optimizer=ensemble_optim, step=jnp.array(0))

    # next_obs, rewards, terms, _ = mbpo.apply(params, obs, acts, True, method='step')

    mbpo.apply(params, offline_replay, 0.2, 512, training_state, wandb, method='train_single_step')

else:
    device = 'cuda'
    offline_replay = OfflineReplay(env, device)

    mbpo = DynamicsEnsemble(
        7, obs_dim, act_dim, [200 for _ in range(4)], 'elu', False, 'normal', 5000,
        True, True, 512, 0.001, 10, 5, None, False, None, None, device,
    )

    train_batch, _ = offline_replay.random_split(0, offline_replay.size)
    train_inputs, _ = mbpo.preprocess_training_batch(train_batch)
    mbpo.scaler.fit(train_inputs)
    mbpo.logger = wandb

    # mbpo.train_single_step(offline_replay, 0.2, 512)
    # Attempting to see if compiling the code speeds up everything...
    batch_size = 512
    validation_ratio = 0.2
    val_size = int(batch_size * validation_ratio)
    train_size = batch_size - val_size
    train_batch, val_batch = offline_replay.random_split(val_size, batch_size * 10)
    mbpo.train_single_step_compiled(train_batch, val_batch)

print(f'Took {time.time() - start} seconds.')


# print(f'O: {next_obs} // {next_obs.shape}')
# print(f'R: {rewards} // {rewards.shape}')
# print(f'D :{terms} // {terms.shape}')

# test = MLPJAX(32, [128, 256, 64], 16, 'relu', True, 'trunc_normal')
# print(test)
#
# testing = []
#
# for _ in range(7):
#     x = jax.random.normal(jax.random.PRNGKey(42), (2, 32))
#     testing.append(jnp.expand_dims(x, 0))
#
# testing = jnp.concatenate(testing, 0)
#
# print(testing.shape)
# testing = testing.mean(0)
# print(testing.shape)
# variables = test.init(jax.random.key(0), x)
#
# for k, v in variables['params'].items():
#     print(k)
#
# output = test.apply(variables, x, moments=False)
# print(output)
# print(len(output))