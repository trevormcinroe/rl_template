import os
import numpy as np
import json
from copy import deepcopy
import tqdm


class DatasetCollector:
    def __init__(self, directory, environment, replay_buffer, environment_name, policy=None):
        self.directory = directory
        self.env = environment
        self.replay_buffer = replay_buffer
        self.env_name = environment_name
        self.policy = policy

        self._n_collect_steps = replay_buffer.capacity
        self.dataset = {}

        self._check_directory()
        self._initialize_dataset_directory()

    def _check_directory(self):
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(f'Given directory does not exist: {self.directory}')

    def _initialize_dataset_directory(self):
        if os.path.isdir(f'{self.directory}/custom_datasets'):
            print(f'Given directory already exist: {self.directory}. /n Do you wish to override? y/n')
            choice = input()

            if np.any([choice.lower() == 'y', choice.lower() == 'yes']):
                os.makedirs(f'{self.directory}/custom_datasets', exist_ok=True)

            else:
                raise FileExistsError(f'Chose to not override {self.directory}/custom_datasets. Exiting.')

        else:
            os.makedirs(f'{self.directory}/custom_datasets')

    def collect(self, n=None):
        if not n:
            if not self.policy:
                self._collect_with_random(self._n_collect_steps)

            else:
                self._collect_with_policy(self._n_collect_steps)

        else:
            if not self.policy:
                self._collect_with_random(self._n_collect_steps)

            else:
                self._collect_with_policy(self._n_collect_steps)

        self._extract_dataset()

        with open(f'{self.directory}/custom_datasets/{self.env_name}.json', 'w') as f:
            json.dump(self.dataset, f)

    def _extract_dataset(self):
        """
        self.dataset = d4rl.qlearning_dataset(env)
        self.states = self.dataset['observations']
        self.actions = self.dataset['actions']
        self.next_states = self.dataset['next_observations']
        self.rewards = self.dataset['rewards'].reshape(-1, 1)
        self.not_dones = (1 - self.dataset['terminals']).reshape(-1, 1)
        self.size = self.rewards.shape[0]
        """
        self.original_dataset = deepcopy(self.replay_buffer)
        self.dataset['observations'] = self.replay_buffer.states.tolist()[:self.replay_buffer.size]
        self.dataset['actions'] = self.replay_buffer.actions.tolist()[:self.replay_buffer.size]
        self.dataset['next_observations'] = self.replay_buffer.next_states.tolist()[:self.replay_buffer.size]
        self.dataset['rewards'] = self.replay_buffer.rewards.tolist()[:self.replay_buffer.size]
        self.dataset['terminals'] = (1 - self.replay_buffer.not_dones).tolist()[:self.replay_buffer.size]

    def _collect_with_policy(self, n_collect_steps):
        with tqdm(total=n_collect_steps) as pbar:
            collected = 0

            while collected <= n_collect_steps:
                obs = self.env.reset()
                done = False

                while not done:
                    a = self.policy(obs)
                    next_obs, reward, done, _ = self.env.step(a)
                    self.replay_buffer.add(obs, a, reward, next_obs, done)
                    collected += 1
                    pbar.update(1)
                    obs = next_obs

    def _collect_with_random(self, n_collect_steps):
        with tqdm(total=n_collect_steps) as pbar:
            collected = 0

            while collected <= n_collect_steps:
                done = False
                obs = self.env.reset()
                done = False

                while not done:
                    a = self.env.action_space.sample()
                    next_obs, reward, done, _ = self.env.step(a)
                    self.replay_buffer.add(obs, a, reward, next_obs, done)
                    collected += 1
                    pbar.update(1)
                    obs = next_obs


