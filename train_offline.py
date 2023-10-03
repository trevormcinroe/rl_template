import gym
from dm_control import suite
import torch
import numpy as np
from networks.ensembles import DynamicsEnsemble
from utils.envs import DMCWrapper
from utils.termination_fns import termination_fns
from utils.replays import ReplayBuffer, OfflineReplay
from rl.sac import SAC
import gym
import d4rl
from ml.classifier import Classifier, DualClassifier
from utils.data import preprocess_sac_batch
import argparse
from tqdm import tqdm
import json
import wandb


args = argparse.ArgumentParser()
args.add_argument('--env')
args.add_argument('--horizon', type=int, default=1)
args.add_argument('--model_file', default=None)
args.add_argument('--plaus_file', default=None)
args.add_argument('--wandb_key')
args.add_argument('--reward_penalty', default=None)
args.add_argument('--reward_penalty_weight', default=1, type=float)
args.add_argument('--loss_penalty', default=None)
args.add_argument('--threshold', default=None, type=float)
args.add_argument('--model_notes', default=None, type=str)
args.add_argument('--eval_model', action='store_true')
args.add_argument('--r', default=0.5, type=float)
args.add_argument('--save_rl', action='store_true')
args = args.parse_args()

"""Environment"""
# env = suite.load('cheetah', 'run')
env = gym.make(args.env)
# env.seed(1)
# env.action_space
# env.action_space.seed(1)
seed = np.random.randint(0, 100000)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# state_dim = sum([env.observation_spec()[x].shape[0] for x in env.observation_spec()])
# action_dim = env.action_spec().shape[0]

# env = DMCWrapper(env, 2)

# eval_env = suite.load('cheetah', 'run')
# eval_env = DMCWrapper(eval_env, 2)

device = 'cuda'

"""Replay"""
env_replay_buffer = ReplayBuffer(100000, state_dim, action_dim, device)

model_retain_epochs = 1
rollout_batch_size = 512
epoch_length = 1000
model_train_freq = 250
rollout_horizon_schedule_min_length = args.horizon
rollout_horizon_schedule_max_length = args.horizon

base_model_buffer_size = int(model_retain_epochs
                            * rollout_batch_size
                            * epoch_length / model_train_freq)
max_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_max_length
min_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_min_length

# Initializing space for the MAX and then setting the pointer ceiling at the MIN
# Doing so lets us scale UP the model horizon rollout length during training
model_replay_buffer = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
model_replay_buffer.max_size = min_model_buffer_size # * 50

"""Model"""
termination_fn = termination_fns[args.env.split('-')[0]]

dynamics_ens = DynamicsEnsemble(
    7, state_dim, action_dim, [200 for _ in range(4)], 'elu', False, 'normal', 5000,
    True, True, 512, 0.001, 10, 5, None, False, args.reward_penalty, args.reward_penalty_weight, None, None, None,
    args.threshold, None, device
)

"""RL"""
# actor, critic, alpha
# orig alpha init=0.1 [0.5]
agent = SAC(
    state_dim, action_dim, [256, 256, 256], 'elu', False, -20, 2, 1e-4, 3e-4,
    3e-4, 0.1, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device
)
n_rl_updates = 40
rl_batch_size = 256
real_ratio = args.r
eval_rl_every = 1000
n_eval_episodes = 10

"""PLAUS"""
if args.plaus_file:
    print(f'Loading plausibility classifier: {args.plaus_file}')
    classifier = DualClassifier(
        state_dim + action_dim + state_dim + dynamics_ens.reward_included,
        [200 for _ in range(2)],
        2,
        'elu',
        1e-3,
        device
    )
    classifier.load_state_dict(
        torch.load(args.plaus_file)
    )
    # loss_penalty = 'Plaus'

else:
    classifier = None
    # loss_penalty = 'None'


"""OFFLINE STUFF!"""
env = gym.make(args.env)
env.reset()
offline_replay = OfflineReplay(env, device)

model_losses = []
disagreement_hist = []
pred_err = []
err_d = {i: [] for i in range(dynamics_ens.n_ensemble_members)}


# NORMING! THIS DOES BOTH STATES AND ACTIONS...
print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

train_batch, _ = offline_replay.random_split(0, offline_replay.size)
train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
dynamics_ens.scaler.fit(train_inputs)

print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

# Need this to be an attribute for dual_classifier reward penalty type (find_neighbors() method)
dynamics_ens.replay = offline_replay

"""LOGGING"""
with open(args.wandb_key, 'r') as f:
    API_KEY = json.load(f)['api_key']

import os
os.environ['WANDB_API_KEY'] = API_KEY
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'

wandb.init(
            project='offline-mbpo',
            # project='testing-wandb',
            entity='trevor-mcinroe',
            name=f'{args.env}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_rp{args.reward_penalty}_rpw{args.reward_penalty_weight}_lp{args.loss_penalty}_t{args.threshold}_{seed}',
        )

dynamics_ens.logger = wandb
agent.logger = wandb


if not args.model_file:
    print('No model file given. Therfore starting model training...')

    for i in tqdm(range(1000000)):
        loss_hist = dynamics_ens.train_single_step(offline_replay, 0.2, 1024)
        model_losses.append(np.mean(loss_hist))

        train_batch, val_batch = offline_replay.random_split(1, 999)
        train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
        train_inputs = dynamics_ens.scaler.transform(train_inputs)
        disagreement, loss, error_dict = dynamics_ens.measure_disagreement(train_inputs, train_targets)
        disagreement_hist.append(disagreement.cpu().item())
        pred_err.append(loss)
        for k, v in error_dict.items():
            err_d[k].append(v)
        #
        # if i % 100 == 0:
        #     print(f'L: {np.mean(model_losses[-100:])}, D: {np.mean(disagreement_hist[-100:])}, E: {np.mean(pred_err[-100:])}')
        # if (i + 1) % 10000 == 0:
        #     torch.save(
        #         dynamics_ens.state_dict(),
        #         f'./models/{args.env}_{i}.pt'
        #     )
        #     #
        #     import pickle
        #     with open(f'./data/{args.env}_model_losses.data', 'wb') as f:
        #         pickle.dump(model_losses, f)
        #
        #     with open(f'./data/{args.env}_disagreement.data', 'wb') as f:
        #         pickle.dump(disagreement_hist, f)
        #
        #     with open(f'./data/{args.env}_pred_err.data', 'wb') as f:
        #         pickle.dump(pred_err, f)

else:
    # Loading the model
    print(f'Loading model file from {args.model_file}')
    dynamics_ens.load_state_dict(torch.load(args.model_file))

    # We place this here (unceremoniously) because the torch.load_state_dict() method will try and load tensors for
    # each parameter, of which the classifier MLP is one...
    dynamics_ens.classifier = classifier

    eval_hist = []
    training_step = 0

    for i in range(500):
        eval_rewards = []

        if args.eval_model:
            model_errors_s = []
            model_errors_r = []

            for _ in range(n_eval_episodes):
                eval_obs = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = agent.act(eval_obs, sample=False)
                    eval_next_obs, reward, done, info = env.step(action)

                    model_input = torch.cat([
                        torch.from_numpy(eval_obs).float().to(dynamics_ens.device).unsqueeze(0),
                        torch.from_numpy(action).float().to(dynamics_ens.device).unsqueeze(0)
                    ], dim=-1)

                    model_input = dynamics_ens.scaler.transform(model_input)

                    with torch.no_grad():
                        model_pred = dynamics_ens.forward_models[
                            np.random.choice(dynamics_ens.selected_elites)
                        ](model_input, moments=False).sample()

                    next_state_pred = model_pred[:, :-1].cpu().numpy() + eval_obs
                    model_errors_s.append(((next_state_pred - eval_next_obs) ** 2).mean())
                    model_errors_r.append(((model_pred[:, -1].cpu().numpy() - reward) ** 2).mean())

                    episode_reward += reward
                    eval_obs = eval_next_obs

                eval_rewards.append(episode_reward)

            wandb.log({
                'model_error_s': np.mean(model_errors_s),
                'model_error_r': np.mean(model_errors_r),
                'step': training_step
            })

        else:
            for _ in range(n_eval_episodes):
                eval_obs = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = agent.act(eval_obs, sample=False)
                    eval_next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    eval_obs = eval_next_obs

                eval_rewards.append(episode_reward)

        eval_hist.append(np.mean(eval_rewards))
        # print(
        #     f'Step: {training_step}, R: {np.mean(eval_rewards)}'
        # )
        wandb.log(
            {'step': training_step, 'eval_returns': np.mean(eval_rewards)}
        )

        for j in tqdm(range(1000)):
            # Need to start with filling the model_replay buffer a small amount
            dynamics_ens.imagine(
                rollout_batch_size,
                args.horizon,
                agent.actor,
                offline_replay,
                model_replay_buffer,
                termination_fn,
                training_step < 0
            )

            # The data used to update the policy is [(1-p)*imagined, p*real]
            agent.update(
                preprocess_sac_batch(offline_replay, model_replay_buffer, rl_batch_size, real_ratio),
                j,
                args.loss_penalty,
                classifier,
                dynamics_ens
            )

            training_step += 1


if args.save_rl:
    agent.save(f'{args.env}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_{seed}')
