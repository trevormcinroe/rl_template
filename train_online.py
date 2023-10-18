import gym
from dm_control import suite
import torch
import d4rl
import numpy as np
from networks.ensembles import DynamicsEnsemble
from utils.envs import DMCWrapper
from utils.replays import ReplayBuffer, resize_model_replay
from utils.data_collection import DatasetCollector
from rl.sac import SAC
from utils.data import preprocess_sac_batch
from utils.termination_fns import termination_fns
import argparse
from tqdm import tqdm
import json
import alternate_envs
import wandb
import dmc2gym
import os


args = argparse.ArgumentParser()
args.add_argument('--wandb_key')
args.add_argument('--model_free', action='store_true')
args.add_argument('--env', type=str)
args.add_argument('--horizon', type=int)
args.add_argument('--imagination_freq', type=int)
args.add_argument('--model_train_freq', type=int)
args.add_argument('--rollout_batch_size', type=int)
args.add_argument('--rl_updates_per', type=int)
args.add_argument('--rl_alpha', type=float, default=0.1)
args.add_argument('--rl_grad_clip', type=float, default=999999999)
args.add_argument('--critic_norm', action='store_true')
args.add_argument('--save_rb', action='store_true')
args.add_argument('--rl_initial_alpha', default=0.1, type=float)
args.add_argument('--n_steps', type=int, default=50000)
args.add_argument('--eval_rl_every', type=int)
args.add_argument('--replay_capacity', type=int)
args.add_argument('--n_seed_steps', type=int, default=1000)
args.add_argument('--n_eval_episodes', type=int, default=10)
args.add_argument('--rl_batch_size', type=int, default=512)
args.add_argument('--real_ratio', type=float, default=0.5)
args = args.parse_args()


"""Environment"""
if not 'dmc2gym' in args.env:
    cutoff = 100_000
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

else:
    cutoff = 100
    if 'walker' in args.env:
        env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=1)
        eval_env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=1)

    elif 'acrobot' in args.env:
        env = dmc2gym.make(domain_name='acrobot', task_name='swingup', from_pixels=False, frame_skip=1)
        eval_env = dmc2gym.make(domain_name='acrobot', task_name='swingup', from_pixels=False, frame_skip=1)

    elif 'manip' in args.env:
        if 'bring_ball' in args.env:
            env = dmc2gym.make(domain_name='manipulator', task_name='bring_ball', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='manipulator', task_name='bring_ball', from_pixels=False, frame_skip=1)

        elif 'bring_peg' in args.env:
            env = dmc2gym.make(domain_name='manipulator', task_name='bring_peg', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='manipulator', task_name='bring_peg', from_pixels=False, frame_skip=1)

        elif 'insert_ball' in args.env:
            env = dmc2gym.make(domain_name='manipulator', task_name='insert_ball', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='manipulator', task_name='insert_ball', from_pixels=False, frame_skip=1)

        elif 'insert_peg' in args.env:
            env = dmc2gym.make(domain_name='manipulator', task_name='insert_peg', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='manipulator', task_name='insert_peg', from_pixels=False, frame_skip=1)

    elif 'hopper' in args.env:
            env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=1)

    elif 'humanoid' in args.env:
        cutoff = 250_000

        if 'cmu' in args.env.lower():
            env = dmc2gym.make(domain_name='humanoid_CMU', task_name='run', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='humanoid_CMU', task_name='run', from_pixels=False, frame_skip=1)

        elif 'walk' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=1)

        elif 'run' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=1)

        elif 'stand' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=1)

    elif 'quadruped' in args.env:
        cutoff = 250_000
        if 'walk' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=1)

        elif 'run' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=1)

        elif 'escape' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=1)

        elif 'fetch' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=1)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=1)

seed = np.random.randint(0, 100000)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = 'cuda'

"""Replay"""
env_replay_buffer = ReplayBuffer(args.replay_capacity, state_dim, action_dim, device)

if not args.model_free:
    model_retain_epochs = 1
    # rollout_batch_size = 512
    epoch_length = 1000
    # args.model_train_freq = 250
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
    model_replay_buffer.max_size = min_model_buffer_size


    def set_rollout_length(args, epoch_step):
        rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                                  / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                                  args.rollout_min_length), args.rollout_max_length))
        return int(rollout_length)

    """Model"""
    if 'humanoid' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
        if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
            dyn_mlp = 800
        else:
            dyn_mlp = 400
    else:
        dyn_mlp = 200

    print(f'Dyn mlp: {dyn_mlp}\n')
    dynamics_ens = DynamicsEnsemble(
        7, state_dim, action_dim, [dyn_mlp for _ in range(4)], 'elu', False, 'normal', 5000,
        True, True, 512, 0.001, 10, 5, None, False, None, 1, None, None, None,
        0, None, device
    )

    try:
        termination_fn = termination_fns[args.env.split('-')[0]]
    except:
        if 'walker' in args.env.lower():
            termination_fn = termination_fns['walker2d']
        else:
            termination_fn = None


print(f'Using termination function: {termination_fn}')

"""RL"""
if 'humanoid' in args.env.lower() or 'ant' in args.env.lower() or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
    if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
        agent_mlp = [1024, 1024, 1024]
    else:
        agent_mlp = [512, 512, 512]
else:
    agent_mlp = [256, 256, 256]

print(f'SAC network size: {agent_mlp}\n')
action_bounds = [env.action_space.low[0], env.action_space.high[0]]
agent = SAC(
    state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
    3e-4, args.rl_initial_alpha, 0.99, 0.005, action_bounds, 256, 2, 2, None, device, args.rl_grad_clip
)


"""Logging"""
with open(args.wandb_key, 'r') as f:
    API_KEY = json.load(f)['api_key']

os.environ['WANDB_API_KEY'] = API_KEY
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'

algo_type = 'MF' if args.model_free else 'MB'

wandb.init(
            project='mujoco-sanity',
            # project='testing-wandb',
            entity='trevor-mcinroe',
            name=f'{args.env}-{algo_type}-rl{args.rl_updates_per}-mtrain{args.model_train_freq}-ifreq{args.imagination_freq}-h{args.horizon}-rbs{args.rollout_batch_size}-{seed}',
        )

if not args.model_free:
    dynamics_ens.logger = wandb

agent.logger = wandb

steps = 0
print(f'Prefilling buffer with {args.n_seed_steps} steps...\n')
while steps < args.n_seed_steps:
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        steps += 1

        env_replay_buffer.add(
            obs, action, reward, next_obs, done
        )

        obs = next_obs

if args.model_free:
    eval_hist = []
    steps = 0
    while steps < args.n_steps:
        done = False
        obs = env.reset()

        while not done:
            action = agent.act(obs, sample=True)
            next_obs, reward, done, info = env.step(action)
            steps += 1

            env_replay_buffer.add(
                obs, action, reward, next_obs, done
            )

            obs = next_obs

            for _ in range(args.rl_updates_per):
                agent.update(
                    preprocess_sac_batch(env_replay_buffer, env_replay_buffer, args.rl_batch_size, args.real_ratio),
                    steps
                )

            if steps % args.eval_rl_every == 0:
                eval_rewards = []

                for _ in range(args.n_eval_episodes):
                    eval_obs = eval_env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action = agent.act(eval_obs, sample=False)
                        eval_next_obs, reward, done, info = eval_env.step(action)
                        episode_reward += reward
                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

                eval_hist.append(np.mean(eval_rewards))
                wandb.log({'step': steps, 'eval_returns': np.mean(eval_rewards)})
                print(
                    f'Step: {steps}, R: {np.mean(eval_rewards)}'
                )

else:
    eval_hist = []
    steps = 0

    if args.n_steps != 50000:
        total_steps_to_train = args.n_steps

    with tqdm(total=total_steps_to_train) as pbar:
        while steps < total_steps_to_train:
            done = False
            obs = env.reset()

            while not done:
                if steps % args.model_train_freq == 0:
                    # print(f'Resuming model training at step {steps}...')
                    # Updating the scaler...
                    train_batch, _ = env_replay_buffer.random_split(0, env_replay_buffer.size)

                    train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
                    dynamics_ens.scaler.fit(train_inputs)

                    loss_ckpt = 999
                    early_stop_ckpt = 5
                    early_stop = 0

                    while early_stop < early_stop_ckpt:
                        loss_hist = dynamics_ens.train_single_step(env_replay_buffer, 0.2, 256)

                        if (loss_ckpt - np.mean(loss_hist)) / loss_ckpt > 0.01:
                            loss_ckpt = np.mean(loss_hist)
                            early_stop = 0
                        else:
                            early_stop += 1

                        wandb.log({
                            'model_early_stop': early_stop,
                            'model_loss': np.mean(loss_hist)
                        })

                    # TODO: this is where we would set the rollout horizon for a non-constant schedule

                    # model_replay_buffer.max_size = base_model_buffer_size * horizon
                    # dynamics_ens.imagine(rollout_batch_size, horizon, agent.actor, env_replay_buffer, model_replay_buffer)

                if steps % args.imagination_freq == 0:
                    dynamics_ens.imagine(
                        args.rollout_batch_size,
                        args.horizon,
                        agent.actor,
                        env_replay_buffer,
                        model_replay_buffer,
                        termination_fn,
                        False
                    )

                    # Rollout think? need to check

                action = agent.act(obs, sample=True)
                next_obs, reward, done, info = env.step(action)

                env_replay_buffer.add(
                    obs, action, reward, next_obs, done
                )

                obs = next_obs

                for _ in range(args.rl_updates_per):
                    agent.update(
                        preprocess_sac_batch(env_replay_buffer, model_replay_buffer, args.rl_batch_size, args.real_ratio),
                        steps
                    )

                if steps % args.eval_rl_every == 0:
                    eval_rewards = []

                    for _ in range(args.n_eval_episodes):
                        eval_obs = eval_env.reset()
                        done = False
                        episode_reward = 0
                        while not done:
                            action = agent.act(eval_obs, sample=False)
                            eval_next_obs, reward, done, info = eval_env.step(action)

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
                    eval_hist.append(np.mean(eval_rewards))
                    wandb.log({'step': steps, 'eval_returns': np.mean(eval_rewards)})
                    # print(
                    #     f'Step: {steps}, R: {np.mean(eval_rewards)}'
                    # )

                steps += 1
                pbar.update(1)
