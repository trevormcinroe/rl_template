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
import numpy as np


import argparse
args = argparse.ArgumentParser()
args.add_argument('--jax', action='store_true')
args = args.parse_args()

"""LOGGING"""
with open('../wandb.txt', 'r') as f:
    API_KEY = json.load(f)['api_key']

import os
os.environ['WANDB_API_KEY'] = API_KEY
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'

wandb.init(
            project='jax-testing',
            entity='trevor-mcinroe',
            name=f'{"jax" if args.jax else "torch"}-hcrandom-mbpo-training',
        )


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

    train_batch, _ = offline_replay.random_split(0, offline_replay.size)
    train_inputs, _ = mbpo.preprocess_training_batch(train_batch)
    mbpo.scaler.fit(jnp.concatenate([obs, acts], -1))

    params = mbpo.init(jax.random.key(0), obs, acts, True)

    opt_state = ensemble_optim.init(params)
    # print(opt_state)

    # TODO: the opt_state needs to be a tuple??
    training_state = TrainingState(params=params, opt_state=opt_state, optimizer=ensemble_optim, step=jnp.array(0))

    # next_obs, rewards, terms, _ = mbpo.apply(params, obs, acts, True, method='step')

    # """TRYING TO JIT FUNCTION!"""
    # batch_size = 512
    # validation_ratio = 0.2
    # val_size = int(batch_size * validation_ratio)
    # train_size = batch_size - val_size
    # train_batch, val_batch = offline_replay.random_split(val_size, batch_size * 10)
    #
    # def _train_mbpo_jax(model, training_state, train_batch, val_batch):
    #     train_inputs, train_targets = model.preprocess_training_batch(train_batch)
    #     val_inputs, val_targets = model.preprocess_training_batch(val_batch)
    #     train_size = train_inputs.shape[0]
    #     train_inputs, val_inputs = model.scaler.transform(train_inputs), model.scaler.transform(val_inputs)
    #
    #     val_loss = [1e5 for _ in range(model.n_ensemble_members)]
    #     epoch = 0
    #     cnt = 0
    #     early_stop = False
    #
    #     idxs = np.random.randint(train_size, size=[model.n_ensemble_members, train_size])
    #
    #     while not early_stop:
    #         for b in range(int(np.ceil(train_size / model.batch_size))):
    #             batch_idxs = idxs[:, b * model.batch_size:(b + 1) * model.batch_size]
    #
    #             # Model forward pass and grad computation
    #             def _mbpo_loss(params, train_inputs, batch_idxs):
    #                 means, logvars = model.apply(
    #                     params, train_inputs, batch_idxs, method='all_forward_models'
    #                 )
    #
    #                 inv_var = jnp.exp(-logvars)
    #                 var_loss = logvars.mean(axis=[1, 2])
    #                 mse_loss = (((means - train_targets[batch_idxs, :]) ** 2) * inv_var).mean(axis=[1, 2])
    #                 loss = (mse_loss + var_loss).sum()
    #                 for i in range(model.n_ensemble_members):
    #                     loss += 0.01 * model.forward_models[i].max_logvar.sum() - 0.01 * model.forward_models[
    #                         i].min_logvar.sum()
    #
    #                 return loss
    #
    #             loss, grads = jax.value_and_grad(_mbpo_loss)(training_state.params, train_inputs, batch_idxs)
    #             updates, new_opt_state = training_state.optimizer.update(grads, training_state.opt_state,
    #                                                                      training_state.params)
    #             new_params = optax.apply_updates(training_state.params, updates)
    #             train_state = TrainingState(params=new_params, opt_state=new_opt_state,
    #                                         optimizer=training_state.optimizer, step=training_state.step + 1)
    #
    #             model.logger.log({'training_loss': loss})
    #
    #         model.shuffle_rows(idxs)
    #         new_val_loss = model.apply(training_state.params, val_inputs, val_targets, None, method='evaluate')
    #         model.logger.log({'eval_loss': jnp.mean(new_val_loss)})
    #         early_stop, val_loss, cnt = model._is_early_stop(val_loss, new_val_loss, cnt)
    #         epoch += 1
    #
    # _train_mbpo_jax(mbpo, training_state, train_batch, val_batch)

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