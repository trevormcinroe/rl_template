import gym
from dm_control import suite
import dmc2gym
import torch
import numpy as np
from networks.ensembles import DynamicsEnsemble
from utils.envs import DMCWrapper
from utils.termination_fns import termination_fns
from utils.replays import ReplayBuffer, OfflineReplay
from rl.sac import SAC
from rl.sac_qens import SACQEns
import gym
import d4rl
from ml.classifier import Classifier, DualClassifier
import alternate_envs
from ml.mixtures import GMM
from ml.ceb import CEB
from utils.data import preprocess_sac_batch, preprocess_sac_batch_oto
from planners.ceb_planners import CEBTreePlanner, CEBParallelPlanner, CEBVecTree
from planners.disagreement_planners import DisagreementVecTree
import argparse
from tqdm import tqdm
import json
import wandb
from copy import deepcopy


args = argparse.ArgumentParser()
args.add_argument('--env')
args.add_argument('--a_repeat', type=int, default=1)
args.add_argument('--custom_filepath', default=None)
args.add_argument('--horizon', type=int, default=1)
args.add_argument('--offline_steps', type=int, required=True)
args.add_argument('--online_steps', type=int, required=True)
args.add_argument('--model_file', default=None)
args.add_argument('--rl_file', default=None)
args.add_argument('--rl_post_online_file', default=None)
args.add_argument('--ceb_file', default=None)
args.add_argument('--ceb_beta', type=float)
args.add_argument('--ceb_z_dim', type=int)
args.add_argument('--ceb_weight', type=float, default=1.)
args.add_argument('--ceb_planner', action='store_true')
args.add_argument('--disagree_planner', action='store_true')
args.add_argument('--ceb_planner_noise', type=float, default=0.0)
args.add_argument('--ceb_width', type=int)
args.add_argument('--ceb_depth', type=int)
args.add_argument('--ceb_update_freq', type=int, default=999999)
args.add_argument('--learned_marginal', action='store_true')
args.add_argument('--act_ceb_pct', type=float)
args.add_argument('--plaus_file', default=None)
args.add_argument('--wandb_key')
args.add_argument('--reward_penalty', default=None)
args.add_argument('--reward_penalty_weight', default=1, type=float)
args.add_argument('--loss_penalty', default=None)
args.add_argument('--threshold', default=None, type=float)
args.add_argument('--model_notes', default=None, type=str)
args.add_argument('--eval_model', action='store_true')
args.add_argument('--r', default=0.5, type=float)
args.add_argument('--imagination_freq', type=int)
args.add_argument('--model_train_freq', type=int)
args.add_argument('--rollout_batch_size', type=int)
args.add_argument('--save_rl_post_online', action='store_true')
args.add_argument('--save_rl_post_offline', action='store_true')
args.add_argument('--rl_updates_per', type=int)
args.add_argument('--rl_grad_clip', type=float, default=999999999)
args.add_argument('--bc_policy', action='store_true')
args.add_argument('--value_ablation', action='store_true')
args.add_argument('--save_rl_bc', action='store_true')
args.add_argument('--bc_file', type=str)
args.add_argument('--copy_full_to_bc', action='store_true')
args.add_argument('--bc_step_pct', type=float)
args.add_argument('--disagreement_weight', type=float, default=1.0)
args.add_argument('--secondary_offline_steps', type=int)
args.add_argument('--act_ucb_pct', type=float)
args.add_argument('--ucb_weight', default=1., type=float)
args.add_argument('--critic_norm', action='store_true')
args.add_argument('--exp_name', type=str, default='oto-mbpo')
args.add_argument('--rl_initial_alpha', default=0.1, type=float)
args.add_argument('--q_ens', action='store_true')
args.add_argument('--wandb_offline', action='store_true')
args.add_argument('--save_trajs', action='store_true')
args.add_argument('--save_trajs_notes', type=str)
args.add_argument('--eval_trajs', action='store_true')
args = args.parse_args()

print(f'CN: {args.critic_norm}')

#
if args.custom_filepath == 'None':
    args.custom_filepath = None

"""Environment"""
if not 'dmc2gym' in args.env:
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

else:
    if 'walker' in args.env:
        env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
        eval_env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

    elif 'hopper' in args.env:
        env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=args.a_repeat)
        eval_env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=args.a_repeat)

    elif 'humanoid' in args.env:
        if 'walk' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

        if 'run' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=args.a_repeat)

        if 'stand' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=args.a_repeat)

    elif 'quadruped' in args.env:
        if 'walk' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

        elif 'run' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=args.a_repeat)

        elif 'escape' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=args.a_repeat)

        elif 'fetch' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=args.a_repeat)

# env.seed(1)
# env.action_space
# env.action_space.seed(1)
seed = np.random.randint(0, 100000)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = 'cuda'

"""Replays"""
online_replay_buffer = ReplayBuffer(100000, state_dim, action_dim, device)

model_retain_epochs = 1
# rollout_batch_size = 512
epoch_length = args.model_train_freq
# model_train_freq = 250
rollout_horizon_schedule_min_length = args.horizon
rollout_horizon_schedule_max_length = args.horizon

base_model_buffer_size = int(model_retain_epochs
                            * args.rollout_batch_size
                            * epoch_length / args.model_train_freq)
max_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_max_length
min_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_min_length

# Initializing space for the MAX and then setting the pointer ceiling at the MIN
# Doing so lets us scale UP the model horizon rollout length during training
model_replay_buffer = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
model_replay_buffer.max_size = min_model_buffer_size # * 50

print(f'Model replay buffer capacity: {max_model_buffer_size}\n')

"""Model"""
termination_fn = termination_fns[args.env.split('-')[0]]
print(f'Using termination function: {termination_fn}')

if 'humanoid' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
    bs = 1024
    if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
        dynamics_hidden = 800
    else:
        dynamics_hidden = 400
else:
    bs = 256
    dynamics_hidden = 200

print(f'Dynamics hidden: {dynamics_hidden}\n')

dynamics_ens = DynamicsEnsemble(
    7, state_dim, action_dim, [dynamics_hidden for _ in range(4)], 'elu', False, 'normal', 5000,
    True, True, 512, 0.001, 10, 5, None, False, args.reward_penalty, args.reward_penalty_weight, None, None, None,
    args.threshold, None, device
)

"""RL"""
if 'humanoid' in args.env.lower() or 'ant' in args.env.lower() or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
    if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
        agent_mlp = [1024, 1024, 1024]
    else:
        agent_mlp = [512, 512, 512]
else:
    agent_mlp = [256, 256, 256]

print(f'Agent mlp: {agent_mlp}\n')
if args.q_ens:
    print('Using an ensemble of q-functions\n')
    agent = SACQEns(
        state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
        3e-4, args.rl_initial_alpha, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device, args.rl_grad_clip
    )

else:
    agent = SAC(
        state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
        3e-4, args.rl_initial_alpha, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device, args.rl_grad_clip
    )

# n_rl_updates = 40
rl_batch_size = 512
real_ratio = args.r
# eval_rl_every = 1000
n_eval_episodes = 10
online_ratio = args.r

"""CEB"""
if args.ceb_file:
    print(f'Loading CEB encoders from: {args.ceb_file}\n')

    try:
        ceb = CEB(state_dim, action_dim, [1024, 512, 512], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
        ceb.load(args.ceb_file)
        print(f'Large encoders: {[1024, 512, 512]}')
    except:
        ceb = CEB(state_dim, action_dim, [256, 128, 64], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
        ceb.load(args.ceb_file)
        print(f'Small encoders: {[256, 128, 64]}')


"""OFFLINE STUFF!"""
# env = gym.make(args.env)
env.reset()
offline_replay = OfflineReplay(env, device, args.custom_filepath)

# NORMING! THIS DOES BOTH STATES AND ACTIONS...
print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

train_batch, _ = offline_replay.random_split(0, offline_replay.size)
train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
dynamics_ens.scaler.fit(train_inputs)

print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

# Need this to be an attribute for dual_classifier reward penalty type (find_neighbors() method)
dynamics_ens.replay = offline_replay

if args.eval_model:
    exp_env = gym.make(args.env.replace('random', 'expert').replace('medium', 'expert'))
    exp_replay = OfflineReplay(exp_env, device)

if args.ceb_planner:
    ceb.scaler = deepcopy(dynamics_ens.scaler)

"""LOGGING"""
with open(args.wandb_key, 'r') as f:
    API_KEY = json.load(f)['api_key']

import os
os.environ['WANDB_API_KEY'] = API_KEY
if not args.wandb_offline:
    os.environ['WANDB_DIR'] = './wandb'
    os.environ['WANDB_CONFIG_DIR'] = './wandb'
    mode = 'online'

else:
    os.environ['WANDB_DIR'] = './offline_wandb'
    os.environ['WANDB_CONFIG_DIR'] = './offline_wandb'
    mode = 'offline'

wandb.init(
            project=args.exp_name,
            entity='trevor-mcinroe',
            mode=mode,
            name=f'{args.env}_a{args.a_repeat}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_online{args.online_steps}_{seed}',
        )

wandb.init()

"""PLANNING"""
if args.ceb_planner:
    """PLANNERS"""
    # planner = CEBTreePlanner()
    planner = CEBVecTree(lambda_q=0.0, lambda_r=1.0, noise_std=args.ceb_planner_noise)
    # planner = CEBParallelPlanner()
    planner.logger = wandb
    planner.termination_fn = termination_fn

if args.disagree_planner:
    planner = DisagreementVecTree(lambda_q=0.0, lambda_r=1.0, noise_std=args.ceb_planner_noise)
    planner.logger = wandb
    planner.termination_fn = termination_fn
    ceb = None

dynamics_ens.logger = wandb
agent.logger = wandb

wandb.log({
    'ceb_planner_noise': args.ceb_planner_noise,
    'ceb_width': args.ceb_width,
    'ceb_depth': args.ceb_depth,
    'ceb_update_freq': args.ceb_update_freq
})

if args.model_file:
    # Loading the model
    if args.model_file == 'online-test':
        print(f'Performing online test. Starting from scratch...\n')
    else:
        print(f'Loading model file from {args.model_file}\n')
        dynamics_ens.load_state_dict(torch.load(args.model_file))

else:
    """Model Fitting"""
    print(f'No model file given. Training model from scratch...\n')
    model_fitting_steps = 0
    loss_ckpt = 999
    early_stop = 250
    early_stop_counter = 0

    while early_stop_counter < early_stop:
        loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, bs)

        batch_size = 1024
        b_idx = 0
        e_idx = b_idx + batch_size
        state_error = []
        reward_error = []

        while e_idx <= dynamics_ens.replay.size:
            state = dynamics_ens.replay.states[b_idx: e_idx]
            action = dynamics_ens.replay.actions[b_idx: e_idx]
            next_state = dynamics_ens.replay.next_states[b_idx: e_idx]
            reward = dynamics_ens.replay.rewards[b_idx: e_idx]
            not_done = dynamics_ens.replay.not_dones[b_idx: e_idx]

            train_batch = (
                torch.FloatTensor(state).to('cuda'),
                torch.FloatTensor(action).to('cuda'),
                torch.FloatTensor(next_state).to('cuda'),
                torch.FloatTensor(reward).to('cuda'),
                torch.FloatTensor(not_done).to('cuda')
            )

            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
            train_inputs = dynamics_ens.scaler.transform(train_inputs)

            with torch.no_grad():
                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                    train_inputs
                )

            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
            state_error.append(state_err)
            reward_error.append(reward_err)

            b_idx += batch_size
            e_idx += batch_size

            if np.all([b_idx < dynamics_ens.replay.size, e_idx > dynamics_ens.replay.size]):
                e_idx = dynamics_ens.replay.size

        curr_loss = np.mean(state_error) + np.mean(reward_error)
        if loss_ckpt > curr_loss:
            loss_ckpt = curr_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # if model_fitting_steps < 1000:
        #     early_stop_counter = 0

        wandb.log({
            'model_early_stop': early_stop_counter,
            'model_loss': np.mean(loss_hist),
            'loss_ckpt': loss_ckpt,
            'curr_loss': curr_loss,
            'model_fitting_steps': model_fitting_steps,
            'state_err': np.mean(state_error),
            'reward_err': np.mean(reward_error)
        })
        model_fitting_steps += 1

        if model_fitting_steps % 1000 == 0:
            if not args.custom_filepath:
                extra = None
            elif 'random' in args.custom_filepath:
                extra = 'random'
            elif 'medium' in args.custom_filepath:
                extra = 'medium'
            elif 'medium-replay' in args.custom_filepath:
                extra = 'medium-replay'
            else:
                extra = None

            print(f'Saving model to: ./models_generalization/{args.env}_a{args.a_repeat}_{extra}_{seed}.pt\n')
            torch.save(
                dynamics_ens.state_dict(),
                f'./models_generalization/{args.env}_a{args.a_repeat}_{extra}_step{model_fitting_steps}_{seed}.pt'
            )

    # Done here
    if not args.custom_filepath:
        extra = None
    elif 'random' in args.custom_filepath:
        extra = 'random'
    elif 'medium' in args.custom_filepath:
        extra = 'medium'
    elif 'medium-replay' in args.custom_filepath:
        extra = 'medium-replay'
    else:
        extra = None

    print(f'Saving model to: ./models_generalization/{args.env}_a{args.a_repeat}_{extra}_{seed}.pt\n')
    torch.save(
        dynamics_ens.state_dict(),
        f'./models_generalization/{args.env}_a{args.a_repeat}_{extra}_step{model_fitting_steps}_{seed}.pt'
    )


"""Offline pre-training"""
eval_hist = []
offline_pretraining_step = 0

if not args.rl_file:
    print(f'No RL file given. Starting policy pre-training from offline dataset...\n')
    while offline_pretraining_step <= args.offline_steps:
        # Eval policy
        eval_rewards = []
        if args.eval_model:
            model_errors_s = []
            model_errors_r = []

            for _ in range(n_eval_episodes):
                eval_obs = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                    eval_next_obs, reward, done, info = env.step(action)
                    wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

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
                'step': offline_pretraining_step
            })

        else:
            for _ in range(n_eval_episodes):
                eval_obs = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                    eval_next_obs, reward, done, info = env.step(action)
                    wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})
                    episode_reward += reward
                    eval_obs = eval_next_obs

                eval_rewards.append(episode_reward)

        wandb.log(
            {'step': offline_pretraining_step, 'eval_returns': np.mean(eval_rewards)}
        )

        # Training loop
        for j in tqdm(range(25000)):
            # Need to start with filling the model_replay buffer a small amount
            dynamics_ens.imagine(
                512,
                args.horizon,
                agent.actor,
                offline_replay,
                model_replay_buffer,
                termination_fn,
                offline_pretraining_step < 0
            )

            # The data used to update the policy is [(1-p)*imagined, p*real]
            agent.update(
                preprocess_sac_batch(offline_replay, model_replay_buffer, rl_batch_size, real_ratio),
                j,
                args.loss_penalty,
                None,
                dynamics_ens
            )

            offline_pretraining_step += 1

    # Saving the RL model post-offline
    if args.save_rl_post_offline:
        print(f'Saving RL file to: ./policies/{args.env}_a{args.a_repeat}-k{args.horizon}_m{args.model_notes}_r{real_ratio}-{seed}-post_offline')
        agent.save(f'./policies/{args.env}_a{args.a_repeat}-k{args.horizon}_m{args.model_notes}_r{real_ratio}-{seed}-post_offline')

else:
    if args.rl_file == 'online-test':
        print(f'Performing an online test, so base agent is randomly initialized\n')
    else:
        print(f'Loading RL file from {args.rl_file}\n')
        agent.load(args.rl_file)

    # Here in the logic flow, we have an empty model replay buffer. This is because we did not perform any offline
    # pre-training. So, let's just fill the model_replay_buffer with a handfull of trajectories using the frozen
    # policy from args.rl_file
    for _ in range(5):
        dynamics_ens.imagine(
            args.rollout_batch_size,
            args.horizon,
            agent.actor,
            offline_replay,
            model_replay_buffer,
            termination_fn,
            offline_pretraining_step < 0
        )

if args.bc_policy:
    bc_agent = SAC(
        state_dim, action_dim, [256, 256, 256], 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
        3e-4, 0.1, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device
    )
    bc_agent.logger = wandb

    if args.bc_file:
        print(f'Loading companion policy from {args.bc_file}\n')
        bc_agent.load(args.bc_file)

    elif args.copy_full_to_bc:
        print('Copying the offline agent weights directly to companion policy\n')
        # Copying over the critics
        bc_agent.critic.load_state_dict(agent.critic.state_dict())
        bc_agent.target_critic.load_state_dict(agent.target_critic.state_dict())
        bc_agent.actor.load_state_dict(agent.actor.state_dict())

    else:
        print('Performing offline training of the companion policy...\n')
        bc_step = 0

        # Copying over the critics
        bc_agent.critic.load_state_dict(agent.critic.state_dict())
        bc_agent.target_critic.load_state_dict(agent.target_critic.state_dict())

        for i in tqdm(range(500_000)):
            # Periodically evaluate the bc agent
            if i % 10000 == 0:
                eval_rewards = []
                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action = bc_agent.act(eval_obs, sample=False)
                        eval_next_obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

                wandb.log(
                    {'bc_step': bc_step, 'bc_eval_returns': np.mean(eval_rewards)}
                )

            batch = preprocess_sac_batch(offline_replay, model_replay_buffer, rl_batch_size, real_ratio)
            obs, action, next_obs, reward, not_done = batch
            # Sampling an action from the agent we wish to BC from
            # with torch.no_grad():
            #     behavior_action = agent.actor(obs).sample().clamp(*agent.action_range)
            bc_agent.update_bc_and_uncertainty(obs, dynamics_ens, args.disagreement_weight)

            # To make sure we have a good representation of the behavior policy's action-selection tendencies,
            # let's re-fill the model_replay_buffer periodically
            if i % args.imagination_freq == 0:
                dynamics_ens.imagine(
                    args.rollout_batch_size,
                    args.horizon,
                    agent.actor,
                    offline_replay,
                    model_replay_buffer,
                    termination_fn,
                    offline_pretraining_step < 0
                )

            bc_step += 1

        if args.save_rl_bc:
            bc_agent.save(
                f'./policies/{args.env}_a{args.a_repeat}-k{args.horizon}_m{args.model_notes}_r{real_ratio}-{seed}-companion'
            )


"""Online fine-tuning phase"""
# Time to finetune online
online_steps = 0

# First, let's get a baseline of eval performance before any training
eval_rewards = []
if args.eval_model:
    model_errors_s = []
    model_errors_r = []

    for _ in range(n_eval_episodes):
        eval_obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, dist = agent.act(eval_obs, sample=False, return_dist=True)
            eval_next_obs, reward, done, info = env.step(action)

            wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

            # Measuring disagreement
            sa = torch.cat(
                [torch.FloatTensor(eval_obs).to(agent.device), torch.FloatTensor(action).to(agent.device)], dim=-1
            )
            oa = dynamics_ens.scaler.transform(sa)
            means = []
            for mem in dynamics_ens.selected_elites:
                mean, _ = dynamics_ens.forward_models[mem](
                    oa
                )
                means.append(mean.unsqueeze(0))

            means = torch.cat(means, dim=0)
            disagreement = (torch.norm(means - means.mean(0), dim=-1)).mean(0).item()
            wandb.log({'disagreement': disagreement})

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

            if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env, 'relocate' in args.env]):
                if info['goal_achieved']:
                    episode_reward = 1
                    done = True
                else:
                    pass
            else:
                episode_reward += reward

            eval_obs = eval_next_obs

        eval_rewards.append(episode_reward)

    wandb.log({
        'model_error_s': np.mean(model_errors_s),
        'model_error_r': np.mean(model_errors_r),
        'step': offline_pretraining_step + online_steps
    })

else:
    for _ in range(n_eval_episodes):
        eval_obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, dist = agent.act(eval_obs, sample=False, return_dist=True)
            eval_next_obs, reward, done, info = env.step(action)
            wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

            # Measuring disagreement
            sa = torch.cat(
                [torch.FloatTensor(eval_obs).to(agent.device), torch.FloatTensor(action).to(agent.device)], dim=-1
            )
            oa = dynamics_ens.scaler.transform(sa)
            means = []
            for mem in dynamics_ens.selected_elites:
                mean, _ = dynamics_ens.forward_models[mem](
                    oa
                )
                means.append(mean.unsqueeze(0))

            means = torch.cat(means, dim=0)
            disagreement = (torch.norm(means - means.mean(0), dim=-1)).mean(0).item()
            wandb.log({'disagreement': disagreement})

            if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env, 'relocate' in args.env]):
                if info['goal_achieved']:
                    episode_reward = 1
                    done = True
                else:
                    pass
            else:
                episode_reward += reward

            eval_obs = eval_next_obs

        eval_rewards.append(episode_reward)

wandb.log(
    {'step': offline_pretraining_step + online_steps,
     'eval_returns': np.mean(eval_rewards)}
)

# (1) Filling the online_replay_buffer with an entire episode for seed steps. We carefully use random actions!
print(f'Prefilling online replay buffer with 1000 random steps...\n')
prefill = 0

while prefill < 1000:
    done = False
    obs = env.reset()

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        online_replay_buffer.add(obs, action, reward, next_obs, done)
        prefill += 1
        obs = next_obs

        online_steps += 1

        if prefill >= 1000:
            break

if args.ceb_file:
    if args.learned_marginal:
        print(f'Fitting GMM marginal m(z)...')
        marginal = GMM(32, args.ceb_z_dim)
        marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)

        for i in tqdm(range(50_000)):
            batch = model_replay_buffer.sample(512, True)
            s, a, *_ = batch
            sa = torch.cat([s, a], dim=-1)
            sa = ceb.scaler.transform(sa)

            z_dist = ceb.e_zx(sa, moments=False)
            z = z_dist.mean

            m_log_prob = marginal.log_prob(z).sum(-1, keepdim=True)
            loss = -m_log_prob.mean()

            marginal_opt.zero_grad()
            loss.backward()
            marginal_opt.step()

        ceb.marginal_z = marginal

    print(f'\nUpdating CEB global rate attribute...')
    # Want to update the global rate using samples from the policy's current occupancy measure,
    # which can be found in the model_replay_buffer
    ceb.update_global_rate(model_replay_buffer, scaler=ceb.scaler)

if args.bc_policy and args.rl_post_online_file is None:
    print(f'Beginning online RL training with a companion policy...\n')
    with tqdm(total=args.online_steps) as pbar:
        while online_steps <= args.online_steps:

            # Now going through training
            done = False
            obs = env.reset()
            episode_steps = 0

            # Picking the policy that will play out the episodes
            # if np.random.rand() < args.bc_step_pct:
            #     interacting_policy = bc_agent
            #     # bc = True
            # else:
            #     interacting_policy = agent
            #     # bc = False

            while not done:
                if np.random.rand() < args.bc_step_pct:
                    interacting_policy = bc_agent
                    # bc = True
                else:
                    interacting_policy = agent
                    # bc = False

                action = interacting_policy.act(obs, sample=True)
                episode_steps += 1
                next_obs, reward, done, info = env.step(action)

                online_replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs

                # Train every step
                if args.value_ablation:
                    for _ in range(args.rl_updates_per):
                        # (1) training \pi
                        agent.update(
                            preprocess_sac_batch_oto(
                                offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio,
                                online_ratio
                            ),
                            online_steps,
                            args.loss_penalty,
                            None,
                            dynamics_ens
                        )

                        # (2) Update companion value
                        batch = preprocess_sac_batch_oto(
                            offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio,
                            online_ratio
                        )
                        bc_obs, bc_action, bc_next_obs, bc_reward, bc_not_done = batch
                        bc_agent.update_critic_companion(
                            bc_obs, bc_action, bc_reward, bc_next_obs, bc_not_done,
                            dynamics_ens, args.disagreement_weight
                        )

                        # (3) Update companion actor
                        if online_steps % bc_agent.actor_update_freq == 0:
                            bc_agent.update_actor_and_alpha(bc_obs)
                            # bc_agent.update_bc_single(bc_obs, dynamics_ens, args.disagreement_weight)

                        if online_steps % bc_agent.critic_target_update_freq == 0:
                            for param, target_param in zip(bc_agent.critic.parameters(), bc_agent.target_critic.parameters()):
                                target_param.data.copy_(bc_agent.tau * param.data + (1.0 - bc_agent.tau) * target_param.data)

                else:
                    for _ in range(args.rl_updates_per):
                        # (1) training \pi
                        agent.update(
                            preprocess_sac_batch_oto(
                                offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio
                            ),
                            online_steps,
                            args.loss_penalty,
                            None,
                            dynamics_ens
                        )

                        # (2) Update companion \pi Q
                        bc_agent.critic.load_state_dict(agent.critic.state_dict())
                        bc_agent.target_critic.load_state_dict(agent.target_critic.state_dict())

                        # (3) Update companion \pi
                        batch = preprocess_sac_batch_oto(
                                offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio
                            )
                        bc_obs, *_ = batch
                        # bc_agent.update_bc_and_uncertainty(bc_obs, dynamics_ens, args.disagreement_weight)
                        # bc_agent.update_bc_single(bc_obs, dynamics_ens, args.disagreement_weight)
                        bc_agent.update_actor_ceb(bc_obs, ceb, args.ceb_weight)
                        # bc_agent.update_bc_and_uncertainty_depth(bc_obs, dynamics_ens, args.disagreement_weight, 50, 5)

                # Train model every N steps:
                if online_steps % args.model_train_freq == 0:
                    # Updating the scaler...
                    train_batch, _ = offline_replay.random_split(0, offline_replay.size)
                    online_batch, _ = online_replay_buffer.random_split(0, online_replay_buffer.size)
                    train_batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
                             zip(train_batch, online_batch)]

                    train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
                    dynamics_ens.scaler.fit(train_inputs)

                    # Now fine-tuning the dynaimcs ensemble
                    loss_ckpt = 999
                    early_stop_ckpt = 5
                    early_stop = 0

                    while early_stop < early_stop_ckpt:
                        loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, bs, online_replay_buffer)

                        # We want to check the model's performance across all fo the data
                        batch_size = 1024
                        b_idx = 0
                        e_idx = b_idx + batch_size
                        state_error = []
                        reward_error = []

                        # First over the offline dataset
                        while e_idx < dynamics_ens.replay.size:
                            state = dynamics_ens.replay.states[b_idx: e_idx]
                            action = dynamics_ens.replay.actions[b_idx: e_idx]
                            next_state = dynamics_ens.replay.next_states[b_idx: e_idx]
                            reward = dynamics_ens.replay.rewards[b_idx: e_idx]
                            not_done = dynamics_ens.replay.not_dones[b_idx: e_idx]

                            train_batch = (
                                torch.FloatTensor(state).to('cuda'),
                                torch.FloatTensor(action).to('cuda'),
                                torch.FloatTensor(next_state).to('cuda'),
                                torch.FloatTensor(reward).to('cuda'),
                                torch.FloatTensor(not_done).to('cuda')
                            )

                            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                            train_inputs = dynamics_ens.scaler.transform(train_inputs)

                            with torch.no_grad():
                                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                    train_inputs
                                )

                            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                            state_error.append(state_err)
                            reward_error.append(reward_err)

                            b_idx += batch_size
                            e_idx += batch_size

                            if np.all([b_idx < dynamics_ens.replay.size, e_idx > dynamics_ens.replay.size]):
                                e_idx = dynamics_ens.replay.size

                        # Next over the online dataset. First need to reset indices
                        b_idx = 0
                        e_idx = b_idx + batch_size

                        while e_idx < online_replay_buffer.size:
                            state = online_replay_buffer.states[b_idx: e_idx]
                            action = online_replay_buffer.actions[b_idx: e_idx]
                            next_state = online_replay_buffer.next_states[b_idx: e_idx]
                            reward = online_replay_buffer.rewards[b_idx: e_idx]
                            not_done = online_replay_buffer.not_dones[b_idx: e_idx]

                            train_batch = (
                                torch.FloatTensor(state).to('cuda'),
                                torch.FloatTensor(action).to('cuda'),
                                torch.FloatTensor(next_state).to('cuda'),
                                torch.FloatTensor(reward).to('cuda'),
                                torch.FloatTensor(not_done).to('cuda')
                            )

                            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                            train_inputs = dynamics_ens.scaler.transform(train_inputs)

                            with torch.no_grad():
                                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                    train_inputs
                                )

                            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                            state_error.append(state_err)
                            reward_error.append(reward_err)

                            b_idx += batch_size
                            e_idx += batch_size

                            if np.all([b_idx < online_replay_buffer.size, e_idx > online_replay_buffer.size]):
                                e_idx = online_replay_buffer.size

                        # Now compare the error across our dataset to previous error checkpoints
                        curr_loss = np.mean(state_error) + np.mean(reward_error)
                        if loss_ckpt > curr_loss:
                            loss_ckpt = curr_loss
                            early_stop = 0
                        else:
                            early_stop += 1

                        # if (loss_ckpt - np.mean(loss_hist)) / loss_ckpt > 0.01:
                        #     loss_ckpt = np.mean(loss_hist)
                        #     early_stop = 0
                        # else:
                        #     early_stop += 1

                        wandb.log({
                            'model_early_stop': early_stop,
                            'model_loss': curr_loss,
                            'step': offline_pretraining_step + online_steps
                        })

                if np.random.rand() < args.bc_step_pct:
                    interacting_policy = bc_agent
                else:
                    interacting_policy = agent

                if online_steps % args.imagination_freq == 0:
                    dynamics_ens.imagine(
                        args.rollout_batch_size,
                        args.horizon,
                        interacting_policy.actor,
                        # agent.actor,
                        online_replay_buffer,
                        model_replay_buffer,
                        termination_fn,
                        (offline_pretraining_step + online_steps) < 0
                    )

                if online_steps % 250 == 0:
                    loss_ckpt = 999
                    early_stop_ckpt = 50
                    early_stop = 0

                    while early_stop < early_stop_ckpt:
                        step_hist = ceb.train_step(256, dynamics_ens.replay, online_replay_buffer)

                        if step_hist['loss'] < loss_ckpt:
                            loss_ckpt = step_hist['loss']
                            early_stop = 0
                        else:
                            early_stop += 1

                        wandb.log({
                            'ceb_early_stop': early_stop,
                            'ceb_loss': step_hist['loss']
                        })

                    ceb.update_global_rate(dynamics_ens.replay, online_replay_buffer)

                online_steps += 1
                pbar.update(1)

            # Eval policy
            eval_rewards = []
            if args.eval_model:

                # First for actual policy
                model_errors_s = []
                model_errors_r = []
                eval_rewards = []

                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = env.step(action)
                        wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

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
                    'step': offline_pretraining_step + online_steps
                })
                wandb.log(
                    {'step': offline_pretraining_step + online_steps,
                     'eval_returns': np.mean(eval_rewards)}
                )

                # Second for bc policy
                model_errors_s = []
                model_errors_r = []
                eval_rewards = []

                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = bc_agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = env.step(action)
                        wandb.log({'companion_actor_entropy': dist.entropy().cpu().mean().item()})

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
                    'bc_model_error_s': np.mean(model_errors_s),
                    'bc_model_error_r': np.mean(model_errors_r),
                    'step': offline_pretraining_step + online_steps
                })
                wandb.log(
                    {'step': offline_pretraining_step + online_steps,
                     'bc_eval_returns': np.mean(eval_rewards)}
                )

            else:
                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = env.step(action)
                        wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})
                        episode_reward += reward
                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

                wandb.log(
                    {'step': offline_pretraining_step + online_steps,
                     'eval_returns': np.mean(eval_rewards)}
                )

elif not args.bc_policy and args.rl_post_online_file is None:
    print(f'Beginning online RL training without a companion policy...\n')
    # Registering a lambda fn
    ucb_pi = lambda obs: agent.act_ucb_batch(obs, 5, dynamics_ens, weight=args.ucb_weight)

    with tqdm(total=args.online_steps) as pbar:
        while online_steps <= args.online_steps:

            # Now going through training
            done = False
            obs = env.reset()
            episode_step = 0
            while not done:
                if args.ceb_planner or args.disagree_planner:
                    if np.random.rand() < args.act_ceb_pct:
                    # if episode_step > 500:
                        action = planner.plan(
                            torch.FloatTensor(obs).unsqueeze(0).to(agent.device),
                            dynamics_ens, ceb, agent.actor, agent.critic, args.ceb_depth, args.ceb_width
                        )
                        # rate = ceb.compute_rate(
                        #     ceb.scaler.transform(torch.cat([
                        #         torch.FloatTensor(obs).unsqueeze(0).to(agent.device),
                        #         torch.FloatTensor(action).unsqueeze(0).to(agent.device)],
                        #         dim=-1)),
                        #     True
                        # )
                        # wandb.log({
                        #     'rate_diff': (rate.mean() - ceb.rate_mean).abs().item(),
                        # })
                    else:
                        action = agent.act(obs, sample=True)
                        # rate = ceb.compute_rate(
                        #     ceb.scaler.transform(torch.cat([
                        #         torch.FloatTensor(obs).unsqueeze(0).to(agent.device),
                        #         torch.FloatTensor(action).unsqueeze(0).to(agent.device)],
                        #         dim=-1)),
                        #     True
                        # )
                        # wandb.log({
                        #     'rate_diff': (rate.mean() - ceb.rate_mean).abs().item(),
                        # })

                    ucb = False

                else:
                    if np.random.rand() < args.act_ucb_pct:
                        action = agent.act_ucb(obs, 5, dynamics_ens, weight=args.ucb_weight)
                        ucb = True
                    else:
                        action = agent.act(obs, sample=True)
                        ucb = False

                next_obs, reward, done, info = env.step(action)
                episode_step += 1

                online_replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs

                # Train every step
                for _ in range(args.rl_updates_per):
                    # 2nd to last arg was originally for classifier. This arg is NOT used for the standard SAC class.
                    # Instead, it handles minibatch sampling for the SACQens class
                    agent.update(
                        preprocess_sac_batch_oto(
                            offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio
                           # online_replay_buffer, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio
                        ),
                        online_steps,
                        args.loss_penalty,
                        [offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio],
                        dynamics_ens
                    )

                online_steps += 1
                pbar.update(1)

                # Train model every N steps:
                if online_steps % args.model_train_freq == 0:
                    # Updating the scaler...
                    train_batch, _ = offline_replay.random_split(0, offline_replay.size)
                    online_batch, _ = online_replay_buffer.random_split(0, online_replay_buffer.size)
                    train_batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
                                   zip(train_batch, online_batch)]

                    train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
                    dynamics_ens.scaler.fit(train_inputs)

                    # TODO: should we also be updating the global rate here? Probably... (see imagination block)
                    loss_ckpt = 999
                    early_stop_ckpt = 5
                    early_stop = 0

                    while early_stop < early_stop_ckpt:
                        loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, bs, online_replay_buffer)
                        # loss_hist = dynamics_ens.train_single_step(online_replay_buffer, 0.2, bs, online_replay_buffer)

                        # We want to check the model's performance across all fo the data
                        batch_size = 1024
                        b_idx = 0
                        e_idx = b_idx + batch_size
                        state_error = []
                        reward_error = []

                        # First over the offline dataset
                        while e_idx <= dynamics_ens.replay.size:
                            state = dynamics_ens.replay.states[b_idx: e_idx]
                            action = dynamics_ens.replay.actions[b_idx: e_idx]
                            next_state = dynamics_ens.replay.next_states[b_idx: e_idx]
                            reward = dynamics_ens.replay.rewards[b_idx: e_idx]
                            not_done = dynamics_ens.replay.not_dones[b_idx: e_idx]

                            train_batch = (
                                torch.FloatTensor(state).to('cuda'),
                                torch.FloatTensor(action).to('cuda'),
                                torch.FloatTensor(next_state).to('cuda'),
                                torch.FloatTensor(reward).to('cuda'),
                                torch.FloatTensor(not_done).to('cuda')
                            )

                            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                            train_inputs = dynamics_ens.scaler.transform(train_inputs)

                            with torch.no_grad():
                                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                    train_inputs
                                )

                            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                            state_error.append(state_err)
                            reward_error.append(reward_err)

                            b_idx += batch_size
                            e_idx += batch_size

                            if np.all([b_idx < dynamics_ens.replay.size, e_idx > dynamics_ens.replay.size]):
                                e_idx = dynamics_ens.replay.size

                        # Next over the online dataset. First need to reset indices
                        b_idx = 0
                        e_idx = b_idx + batch_size

                        while e_idx <= online_replay_buffer.size:
                            state = online_replay_buffer.states[b_idx: e_idx]
                            action = online_replay_buffer.actions[b_idx: e_idx]
                            next_state = online_replay_buffer.next_states[b_idx: e_idx]
                            reward = online_replay_buffer.rewards[b_idx: e_idx]
                            not_done = online_replay_buffer.not_dones[b_idx: e_idx]

                            train_batch = (
                                torch.FloatTensor(state).to('cuda'),
                                torch.FloatTensor(action).to('cuda'),
                                torch.FloatTensor(next_state).to('cuda'),
                                torch.FloatTensor(reward).to('cuda'),
                                torch.FloatTensor(not_done).to('cuda')
                            )

                            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                            train_inputs = dynamics_ens.scaler.transform(train_inputs)

                            with torch.no_grad():
                                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                    train_inputs
                                )

                            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                            state_error.append(state_err)
                            reward_error.append(reward_err)

                            b_idx += batch_size
                            e_idx += batch_size

                            if np.all([b_idx < online_replay_buffer.size, e_idx > online_replay_buffer.size]):
                                e_idx = online_replay_buffer.size

                        # Now compare the error across our dataset to previous error checkpoints
                        curr_loss = np.mean(state_error) + np.mean(reward_error)
                        if loss_ckpt > curr_loss:
                            loss_ckpt = curr_loss
                            early_stop = 0
                        else:
                            early_stop += 1

                        # if (loss_ckpt - np.mean(loss_hist)) / loss_ckpt > 0.01:
                        #     loss_ckpt = np.mean(loss_hist)
                        #     early_stop = 0
                        # else:
                        #     early_stop += 1

                        wandb.log({
                            'model_early_stop': early_stop,
                            'model_loss': curr_loss,
                            'step': offline_pretraining_step + online_steps
                        })

                if online_steps % args.imagination_freq == 0:
                    dynamics_ens.imagine(
                        args.rollout_batch_size,
                        args.horizon,
                        agent.actor,
                        online_replay_buffer,
                        model_replay_buffer,
                        termination_fn,
                        ucb_pi if ucb else False
                    )

                    # ceb.update_global_rate(model_replay_buffer)

                if online_steps % args.ceb_update_freq == 0:
                    # Re-loading the model replay buffer and then fitting CEB-style encoders
                    dynamics_ens.imagine(
                        args.rollout_batch_size,
                        args.horizon,
                        agent.actor,
                        online_replay_buffer,
                        model_replay_buffer,
                        termination_fn,
                        ucb_pi if ucb else False
                    )

                    online_batch, _ = model_replay_buffer.random_split(0, model_replay_buffer.size)
                    s, a, *_ = online_batch
                    sa = torch.cat([s, a], dim=-1)

                    ceb = CEB(state_dim, action_dim, [256, 128, 64], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
                    ceb.scaler = deepcopy(dynamics_ens.scaler)
                    ceb.scaler.fit(sa)

                    # TODO: how many steps should this be done for?
                    for _ in range(50_000):
                        ceb_step_hist = ceb.train_step(512, model_replay_buffer, scaler=ceb.scaler)
                        wandb.log(ceb_step_hist)

                    if args.learned_marginal:
                        # Fitting the marginal
                        marginal = GMM(32, args.ceb_z_dim)
                        marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)

                        for i in range(30_000):
                            batch = model_replay_buffer.sample(512, True)
                            s, a, *_ = batch
                            sa = torch.cat([s, a], dim=-1)
                            sa = ceb.scaler.transform(sa)

                            z_dist = ceb.e_zx(sa, moments=False)
                            z = z_dist.mean

                            m_log_prob = marginal.log_prob(z).sum(-1, keepdim=True)
                            loss = -m_log_prob.mean()

                            marginal_opt.zero_grad()
                            loss.backward()
                            marginal_opt.step()

                        ceb.marginal_z = marginal

                    ceb.update_global_rate(model_replay_buffer, scaler=ceb.scaler)

                    # if args.ceb_file:
                    #     loss_ckpt = 999
                    #     early_stop_ckpt = 2500
                    #     early_stop = 0
                    #
                    #     while early_stop < early_stop_ckpt:
                    #         step_hist = ceb.train_step(512, dynamics_ens.replay, online_replay_buffer)
                    #
                    #         if step_hist['loss'] < loss_ckpt:
                    #             loss_ckpt = step_hist['loss']
                    #             early_stop = 0
                    #         else:
                    #             early_stop += 1
                    #
                    #         wandb.log({
                    #             'ceb_early_stop': early_stop,
                    #             'ceb_loss': step_hist['loss']
                    #         })
                    #
                    #     ceb.update_global_rate(dynamics_ens.replay, online_replay_buffer)

                if online_steps % 1000 == 0:
                    # Eval policy
                    eval_rewards = []
                    if args.eval_model:

                        model_errors_s = []
                        model_errors_r = []
                        rl_error_q = []
                        rl_error_logpi = []
                        ceb_error_mu = []

                        batch_size = 1024
                        b_idx = 0
                        e_idx = b_idx + batch_size

                        while e_idx < exp_replay.size:
                            state = exp_replay.states[b_idx: e_idx]
                            action = exp_replay.actions[b_idx: e_idx]
                            next_state = exp_replay.next_states[b_idx: e_idx]
                            reward = exp_replay.rewards[b_idx: e_idx]
                            not_done = exp_replay.not_dones[b_idx: e_idx]

                            train_batch = (
                                torch.FloatTensor(state).to('cuda'),
                                torch.FloatTensor(action).to('cuda'),
                                torch.FloatTensor(next_state).to('cuda'),
                                torch.FloatTensor(reward).to('cuda'),
                                torch.FloatTensor(not_done).to('cuda')
                            )
                            s, a, ns, r, d = train_batch

                            # Model errors
                            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                            train_inputs = dynamics_ens.scaler.transform(train_inputs)

                            with torch.no_grad():
                                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                    train_inputs
                                )

                            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                            reward_err = (means - train_targets)[:, -1].pow(1).mean().cpu().item()
                            model_errors_s.append(state_err)
                            model_errors_r.append(reward_err)

                            # RL errors
                            q1, q2 = agent.critic(torch.cat([s, a], dim=-1))
                            q_val = torch.min(q1, q2)
                            rl_error_q.append(q_val.mean().cpu().item())

                            dist = agent.actor(s)
                            log_probs = dist.log_prob(a).sum(-1, keepdim=True)
                            rl_error_logpi.append(log_probs.mean().cpu().item())

                            # CEB rate
                            rate = ceb.compute_rate(ceb.scaler.transform(torch.cat([s, a], dim=-1)), True)
                            ceb_error_mu.append((rate - ceb.rate_mean).abs().mean().cpu().item())

                            b_idx += batch_size
                            e_idx += batch_size

                            if np.all([b_idx < exp_replay.size, e_idx > exp_replay.size]):
                                e_idx = exp_replay.size

                        wandb.log({
                            'meval/wm_s': np.mean(model_errors_s),
                            'meval/wm_r': np.mean(model_errors_r),
                            'meval/rl_q': np.mean(rl_error_q),
                            'meval/rl_pi': np.mean(rl_error_logpi),
                            'meval/ceb_r': np.mean(ceb_error_mu),
                            'step': offline_pretraining_step + online_steps
                        })

                    for _ in range(n_eval_episodes):
                        eval_obs = eval_env.reset()
                        done = False
                        episode_reward = 0
                        while not done:
                            action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                            eval_next_obs, reward, done, info = eval_env.step(action)
                            wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

                            # Measuring disagreement
                            sa = torch.cat(
                                [torch.FloatTensor(eval_obs).to(agent.device),
                                 torch.FloatTensor(action).to(agent.device)], dim=-1
                            )
                            oa = dynamics_ens.scaler.transform(sa)
                            means = []
                            for mem in dynamics_ens.selected_elites:
                                mean, _ = dynamics_ens.forward_models[mem](
                                    oa
                                )
                                means.append(mean.unsqueeze(0))

                            means = torch.cat(means, dim=0)
                            disagreement = (torch.norm(means - means.mean(0), dim=-1)).mean(0).item()
                            wandb.log({'disagreement': disagreement})

                            if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env,
                                       'relocate' in args.env]):
                                if info['goal_achieved']:
                                    episode_reward = 1
                                    done = True
                                else:
                                    pass
                            else:
                                episode_reward += reward

                            eval_obs = eval_next_obs

                        eval_rewards.append(episode_reward)

                    wandb.log(
                        {'step': offline_pretraining_step + online_steps,
                         'eval_returns': np.mean(eval_rewards)}
                    )

if args.save_rl_post_online:
    agent.save(f'{args.env}_a{args.a_repeat}_bc{args.bc_policy}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_online{args.online_steps}_{seed}')

if args.rl_post_online_file:
    agent.load(args.rl_post_online_file)

"""Some policy comparing"""
if args.save_trajs:
    store_traj_returns = []
    traj_data = {
        'states': [],
        'actions': [],
        'rewards': []
    }
    for _ in range(10):
        eval_obs = env.reset()
        done = False
        episode_reward = 0
        ep_states, ep_actions, ep_rewards = [], [], []
        while not done:
            action = agent.act(eval_obs, sample=False)
            eval_next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            ep_states.append(eval_obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            eval_obs = eval_next_obs

        store_traj_returns.append(episode_reward)
        traj_data['states'].append(ep_states)
        traj_data['actions'].append(ep_actions)
        traj_data['rewards'].append(ep_rewards)
    print(f'Collected Traj Returns: {np.mean(store_traj_returns)} +- {np.std(store_traj_returns)}')

    import pickle
    with open(f'./{args.save_trajs_notes}.data', 'wb') as f:
        pickle.dump(traj_data, f)


if args.eval_trajs:
    import pickle
    with open(f'./{args.save_trajs_notes}.data', 'rb') as f:
        traj_data = pickle.load(f)

    # (1) Collect an eval traj with the current agent
    eval_obs = env.reset()
    done = False
    episode_reward = 0
    ep_states, ep_actions, ep_rewards = [], [], []
    while not done:
        action = agent.act(eval_obs, sample=False)
        eval_next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        ep_states.append(eval_obs)
        ep_actions.append(action)
        ep_rewards.append(reward)
        eval_obs = eval_next_obs

    print(f'Generated return: {episode_reward}')

    # (2) See how the agent's current Q-fn values the states in the trajectory
    self_qs = []
    for i in range(len(ep_states)):
        sa = torch.cat([
            torch.FloatTensor(ep_states[i]).unsqueeze(0), torch.FloatTensor(ep_actions[i]).unsqueeze(0)
        ], dim=-1).to(device)

        with torch.no_grad():
            q1, q2 = agent.critic(sa)
        q = torch.min(q1, q2)
        self_qs.append(q.cpu().numpy().item())

    # (3) See how the agent's current Q-fn values the states from the loaded file
    other_qs = []
    for i in range(10):
        for j in range(len(traj_data['states'][0])):
            sa = torch.cat([
                torch.FloatTensor(traj_data['states'][i][j]).unsqueeze(0), torch.FloatTensor(traj_data['actions'][i][j]).unsqueeze(0)
            ], dim=-1).to(device)

            with torch.no_grad():
                q1, q2 = agent.critic(sa)
            q = torch.min(q1, q2)
            other_qs.append(q.cpu().numpy().item())

    print(f'Self Qs: {np.mean(self_qs)} +- {np.std(self_qs)}')
    print(f'Other Qs: {np.mean(other_qs)} +- {np.std(other_qs)}')

"""Returning to offline training after we have collected some data online"""
if args.secondary_offline_steps > 0:
    print(f'Fitting the world model one last time, post-online data collection...')
    # Updating the scaler...
    train_batch, _ = offline_replay.random_split(0, offline_replay.size)
    online_batch, _ = online_replay_buffer.random_split(0, online_replay_buffer.size)
    train_batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
                   zip(train_batch, online_batch)]

    train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
    dynamics_ens.scaler.fit(train_inputs)

    loss_ckpt = 999
    early_stop_ckpt = 10
    early_stop = 0

    while early_stop < early_stop_ckpt:
        loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, 256, online_replay_buffer)

        if (loss_ckpt - np.mean(loss_hist)) / loss_ckpt > 0.01:
            loss_ckpt = np.mean(loss_hist)
            early_stop = 0
        else:
            early_stop += 1

        wandb.log({
            'model_early_stop': early_stop,
            'model_loss': np.mean(loss_hist),
            'step': offline_pretraining_step + online_steps
        })

offline_steps = 0
print(f'Starting secondary offline-RL agent training...')
while offline_steps < args.secondary_offline_steps:
    # Eval policy
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
            'step': offline_pretraining_step + online_steps + offline_steps
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

    wandb.log(
        {'step': offline_pretraining_step + online_steps + offline_steps,
         'eval_returns': np.mean(eval_rewards)}
    )

    # Training loop
    for j in tqdm(range(1000)):
        # Need to start with filling the model_replay buffer a small amount
        # 50% for s_0 ~ Offline / s_0 ~ Online
        if np.random.rand() <= 0.5:
            dynamics_ens.imagine(
                512,
                args.horizon,
                agent.actor,
                offline_replay,
                model_replay_buffer,
                termination_fn,
                offline_pretraining_step < 0
            )
        else:
            dynamics_ens.imagine(
                512,
                args.horizon,
                agent.actor,
                online_replay_buffer,
                model_replay_buffer,
                termination_fn,
                offline_pretraining_step < 0
            )

        # The data used to update the policy is [(1-p)*imagined, p*real]
        agent.update(
            preprocess_sac_batch_oto(
                offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio, online_ratio
            ),
            j,
            args.loss_penalty,
            None,
            dynamics_ens
        )

        offline_steps += 1
