import json
import numpy as np
import torch
from torch import FloatTensor
from typing import Tuple, Union, List
from gym import Env
from utils.scalers import StandardScaler
from jax import numpy as jnp
from jax._src.typing import Array


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.capacity = capacity
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.states = np.empty((capacity, obs_dim))
        self.next_states = np.empty((capacity, obs_dim))
        self.actions = np.empty((capacity, action_dim))
        self.rewards = np.empty((capacity, 1))
        self.not_dones = np.empty((capacity, 1))

        self.pointer = 0
        self.size = 0

    def add(self, obs: np.array, action: np.array, reward: float, next_obs: np.array, not_done: bool) -> None:
        np.copyto(self.states[self.pointer], obs)
        np.copyto(self.actions[self.pointer], action)
        np.copyto(self.rewards[self.pointer], reward)
        np.copyto(self.next_states[self.pointer], next_obs)
        np.copyto(self.not_dones[self.pointer], 1 - not_done)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obses: np.array, actions: np.array, rewards: np.array, next_obses: np.array,
                  not_dones: np.array) -> None:
        for obs, action, reward, next_obs, not_done in zip(obses, actions, rewards, next_obses, not_dones):
            self.add(obs, action, reward, next_obs, not_done)

    def sample(
            self, batch_size: int, rl: bool = False
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
        # print(f'INDS!: {ind}')
            ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.not_dones[ind]).to(self.device)
        )

    def sample_traj(
            self, batch_size: int, episode_length: int, horizon: int
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        eoo = np.where(self.not_dones[:self.size] == 0)[0]

        # Getting indexes where eoo != 0
        # [N, T] e.g., (N, 500) for when action_repeat = 2
        # traj_slices selects the beginning step in the subtrajectory
        traj_slices = np.random.choice(episode_length - horizon,
                                       size=len(eoo),
                                       replace=False)

        indexes = np.arange(self.size)

        indexes = np.array([
            indexes[eoo[i] + 1: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon] if i > 0
            else indexes[eoo[i]: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon]
            for i in range(len(eoo) - 1)
        ])

        batch_idxs = np.random.choice(indexes.shape[0], batch_size)
        batch_idxs = [indexes[batch_idxs]]

        training_batch = (
            torch.FloatTensor(self.states[batch_idxs]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs]).to(self.device)
        )

        return training_batch

    def get_all(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        return (
            self.states[:self.pointer],
            self.actions[:self.pointer],
            self.next_states[:self.pointer],
            self.rewards[:self.pointer],
            self.not_dones[:self.pointer]
        )

    def random_split(
            self, val_size: int, batch_size: Union[int, None] = None
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa

        if batch_size is not None:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
            )[:batch_size]

            training_batch = (
                torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
            )

            validation_batch = (
                torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
            )

        else:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
            )

            training_batch = (
                torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
            )

            validation_batch = (
                torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
            )

        return training_batch, validation_batch

    @property
    def n_traj(self) -> np.array:
        return np.where(self.not_dones[:self.size] == 0)[0].shape[0]

    def random_split_traj(
            self, val_size: int, horizon: int, episode_length: int
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa
        """"""
        # TODO: what should we expect to happend when the pointer loops around?
        # First, find end of episode dones
        eoo = np.where(self.not_dones[:self.size] == 0)[0]

        # Getting indexes where eoo != 0
        # [N, T] e.g., (N, 500) for when action_repeat = 2
        # traj_slices selects the beginning step in the subtrajectory
        traj_slices = np.random.choice(episode_length - horizon,
                                       size=len(eoo),
                                       replace=False)

        indexes = np.arange(self.size)

        indexes = np.array([
            indexes[eoo[i] + 1: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon] if i > 0
            else indexes[eoo[i]: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon]
            for i in range(len(eoo) - 1)
        ])

        # We need to now shuffle to ensure a good mixture. Otherwise, val_traj will always
        # be from later-collected trajectories. The below function performs in-place shuffling along the
        # 0th axis, which is what we want
        np.random.shuffle(indexes)

        # TODO: check if these are splitting the data properly -- should be based on JNB tests
        training_batch = (
            torch.FloatTensor(self.states[indexes[val_size:]]).to(self.device),
            torch.FloatTensor(self.actions[indexes[val_size:]]).to(self.device),
            torch.FloatTensor(self.next_states[indexes[val_size:]]).to(self.device),
            torch.FloatTensor(self.rewards[indexes[val_size:]]).to(self.device),
            torch.FloatTensor(self.not_dones[indexes[val_size:]]).to(self.device)
        )

        validation_batch = (
            torch.FloatTensor(self.states[indexes[:val_size]]).to(self.device),
            torch.FloatTensor(self.actions[indexes[:val_size]]).to(self.device),
            torch.FloatTensor(self.next_states[indexes[:val_size]]).to(self.device),
            torch.FloatTensor(self.rewards[indexes[:val_size]]).to(self.device),
            torch.FloatTensor(self.not_dones[indexes[:val_size]]).to(self.device)
        )

        return training_batch, validation_batch


class OfflineReplay:
    def __init__(self, env: Env, device: str, custom_filepath: Union[str, None] = None) -> None:
        import d4rl
        self.env = env
        self.device = device

        if custom_filepath:
            with open(custom_filepath, 'r') as f:
                self.dataset = json.load(f)
                self.states = np.array(self.dataset['observations'])
                self.actions = np.array(self.dataset['actions'])
                self.next_states = np.array(self.dataset['next_observations'])
                self.rewards = np.array(self.dataset['rewards']).reshape(-1, 1)
                self.not_dones = (1 - np.array(self.dataset['terminals'])).reshape(-1, 1)
        else:
            self.dataset = d4rl.qlearning_dataset(env)
            self.states = self.dataset['observations']
            self.actions = self.dataset['actions']
            self.next_states = self.dataset['next_observations']
            self.rewards = self.dataset['rewards'].reshape(-1, 1)
            self.not_dones = (1 - self.dataset['terminals']).reshape(-1, 1)

        self.size = self.rewards.shape[0]

        # The current d4rl datasets are very annoying. For CR, there is no early-termination condition (unlike WW and
        # hopper). Therefore, the below is ONLY FOR CR!~!!!!!@
        self.traj_indices = []

        b_idx = 0
        e_idx = 999

        while e_idx <= self.rewards.shape[0]:
            self.traj_indices.append([b_idx, e_idx])
            b_idx += 1000
            e_idx += 1000

    def random_split(
            self, val_size: int, batch_size: int
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa
        # batch_idxs = np.random.permutation(self.size)[:batch_size]
        batch_idxs = np.random.permutation(
            np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
        )[:batch_size]

        training_batch = (
            torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
        )

        validation_batch = (
            torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
        )

        return training_batch, validation_batch

    def sample(
            self, batch_size: int, rl: bool = False
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        # print(f'INDS!: {ind}')
        # ind = np.random.randint(0, self.size, size=batch_size)[self.not_dones == 1]

        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.not_dones[ind]).to(self.device)
        )

    def random_split_transformer(
            self, val_size: int, batch_size: int, h: int, scaler: StandardScaler
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa
        input_batch_s = []
        input_batch_a = []
        target_batch_s = []
        target_batch_r = []
        ts = []

        # TODO: the below routine only works for trajectories that have no early termination conditions.
        # Perhaps we need to also pre-compute and store some information on each trajectories length?
        for i in range(batch_size):
            traj_index = self.traj_indices[np.random.choice(range(len(self.traj_indices)))]
            traj_end = np.random.choice(np.arange(1, 999))
            traj_beginning = traj_end - h

            traj_index_end = traj_index[1] - (999 - traj_end)

            # We will zero-pad (left) trajectories that are before the horizon length
            if traj_beginning < 0:
                traj_beginning = 0

                num_pads = h - traj_end

                traj_index_beginning = traj_index_end - (h - num_pads)

                # [B, object_dim]
                s = torch.from_numpy(self.states[traj_index_beginning: traj_index_end])
                a = torch.from_numpy(self.actions[traj_index_beginning: traj_index_end])

                # [B, |s| + |a|] --> scale and split out
                sa = torch.cat([s, a], dim=-1)
                sa = scaler.transform(sa.to(self.device))
                s = sa[:, :self.states.shape[-1]]
                a = sa[:, self.states.shape[-1]:]

                s = torch.cat([torch.zeros(num_pads, self.states.shape[-1]).to(self.device), s], dim=0)
                a = torch.cat([torch.zeros(num_pads, self.actions.shape[-1]).to(self.device), a], dim=0)

                t = torch.arange(start=0, end=h, step=1)

            else:
                traj_index_beginning = traj_index_end - h

                # [B, object_dim]
                s = torch.from_numpy(self.states[traj_index_beginning: traj_index_end])
                a = torch.from_numpy(self.actions[traj_index_beginning: traj_index_end])

                # [B, |s| + |a|] -->
                sa = torch.cat([s, a], dim=-1)
                sa = scaler.transform(sa.to(self.device))
                s = sa[:, :self.states.shape[-1]]
                a = sa[:, self.states.shape[-1]:]

                t = torch.arange(start=traj_beginning, end=traj_end, step=1)

            input_batch_s.append(s.unsqueeze(0))
            input_batch_a.append(a.unsqueeze(0))

            target_batch_s.append(
                torch.from_numpy(
                    self.states[traj_index_end] - self.states[traj_index_end - 1])
            )

            target_batch_r.append(
                torch.from_numpy(self.rewards[traj_index_end - 1])
            )

            ts.append(t)

        input_batch_s = torch.vstack(input_batch_s).to(self.device)
        input_batch_a = torch.vstack(input_batch_a).to(self.device)
        target_batch_s = torch.vstack(target_batch_s).to(self.device)
        target_batch_r = torch.vstack(target_batch_r).to(self.device)
        ts = torch.vstack(ts).to(self.device)

        train_batch = (
            input_batch_s[val_size:],
            input_batch_a[val_size:],
            target_batch_s[val_size:],
            target_batch_r[val_size:],
            ts[val_size:]
        )

        val_batch = (
            input_batch_s[:val_size],
            input_batch_a[:val_size],
            target_batch_s[:val_size],
            target_batch_r[:val_size],
            ts[:val_size]
        )

        return train_batch, val_batch


def resize_model_replay(rollout_batch_size, model_train_frequency, horizon, model_replay, model_retain_length=1):
    raise NotImplementedError('This function is not yet implemented correctly.')
    # TODO: 1000 is hardcoded... I'm not certain what it's actually trying to capture. It's 1k for all envs in MBPO paper
    # if 512 * 1000 / 250 -> 2048
    rollouts_per_epoch = rollout_batch_size * 1000 / model_train_frequency
    model_steps_per_epoch = int(horizon * rollouts_per_epoch)
    new_replay_size = model_retain_length * model_steps_per_epoch

    # We don't want to completely lose our entire replay buffer at the moment, so let's just insert one into the other!
    previous_trajectories = model_replay.get_all()
    new_replay = ReplayBuffer(new_replay_size, model_replay.obs_dim, model_replay.action_dim, model_replay.device)
    new_replay.add_batch(*previous_trajectories)
    return new_replay


class ReplayBufferJAX:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.capacity = capacity
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.states = np.empty((capacity, obs_dim))
        self.next_states = np.empty((capacity, obs_dim))
        self.actions = np.empty((capacity, action_dim))
        self.rewards = np.empty((capacity, 1))
        self.not_dones = np.empty((capacity, 1))

        self.pointer = 0
        self.size = 0

    def add(self, obs: np.array, action: np.array, reward: float, next_obs: np.array, not_done: bool) -> None:
        np.copyto(self.states[self.pointer], obs)
        np.copyto(self.actions[self.pointer], action)
        np.copyto(self.rewards[self.pointer], reward)
        np.copyto(self.next_states[self.pointer], next_obs)
        np.copyto(self.not_dones[self.pointer], 1 - not_done)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obses: np.array, actions: np.array, rewards: np.array, next_obses: np.array,
                  not_dones: np.array) -> None:
        for obs, action, reward, next_obs, not_done in zip(obses, actions, rewards, next_obses, not_dones):
            self.add(obs, action, reward, next_obs, not_done)

    def sample(
            self, batch_size: int, rl: bool = False
    ):
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
        # print(f'INDS!: {ind}')
            ind = np.random.randint(0, self.size, size=batch_size)

        return (
            jnp.array(self.states[ind]),
            jnp.array(self.actions[ind]),
            jnp.array(self.next_states[ind]),
            jnp.array(self.rewards[ind]),
            jnp.array(self.not_dones[ind])
        )

    def get_all(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        return (
            self.states[:self.pointer],
            self.actions[:self.pointer],
            self.next_states[:self.pointer],
            self.rewards[:self.pointer],
            self.not_dones[:self.pointer]
        )

    def random_split(
            self, val_size: int, batch_size: Union[int, None] = None
    ):  # noqa

        if batch_size is not None:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
            )[:batch_size]

            training_batch = (
                jnp.array(self.states[batch_idxs[val_size:]]),
                jnp.array(self.actions[batch_idxs[val_size:]]),
                jnp.array(self.next_states[batch_idxs[val_size:]]),
                jnp.array(self.rewards[batch_idxs[val_size:]]),
                jnp.array(self.not_dones[batch_idxs[val_size:]])
            )

            validation_batch = (
                jnp.array(self.states[batch_idxs[:val_size]]),
                jnp.array(self.actions[batch_idxs[:val_size]]),
                jnp.array(self.next_states[batch_idxs[:val_size]]),
                jnp.array(self.rewards[batch_idxs[:val_size]]),
                jnp.array(self.not_dones[batch_idxs[:val_size]])
            )

        else:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
            )

            training_batch = (
                jnp.array(self.states[batch_idxs[val_size:]]),
                jnp.array(self.actions[batch_idxs[val_size:]]),
                jnp.array(self.next_states[batch_idxs[val_size:]]),
                jnp.array(self.rewards[batch_idxs[val_size:]]),
                jnp.array(self.not_dones[batch_idxs[val_size:]])
            )

            validation_batch = (
                jnp.array(self.states[batch_idxs[:val_size]]),
                jnp.array(self.actions[batch_idxs[:val_size]]),
                jnp.array(self.next_states[batch_idxs[:val_size]]),
                jnp.array(self.rewards[batch_idxs[:val_size]]),
                jnp.array(self.not_dones[batch_idxs[:val_size]])
            )

        return training_batch, validation_batch

    @property
    def n_traj(self) -> np.array:
        return np.where(self.not_dones[:self.size] == 0)[0].shape[0]

    def random_split_traj(
            self, val_size: int, horizon: int, episode_length: int
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa
        """"""
        # TODO: what should we expect to happend when the pointer loops around?
        # First, find end of episode dones
        eoo = np.where(self.not_dones[:self.size] == 0)[0]

        # Getting indexes where eoo != 0
        # [N, T] e.g., (N, 500) for when action_repeat = 2
        # traj_slices selects the beginning step in the subtrajectory
        traj_slices = np.random.choice(episode_length - horizon,
                                       size=len(eoo),
                                       replace=False)

        indexes = np.arange(self.size)

        indexes = np.array([
            indexes[eoo[i] + 1: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon] if i > 0
            else indexes[eoo[i]: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon]
            for i in range(len(eoo) - 1)
        ])

        # We need to now shuffle to ensure a good mixture. Otherwise, val_traj will always
        # be from later-collected trajectories. The below function performs in-place shuffling along the
        # 0th axis, which is what we want
        np.random.shuffle(indexes)

        # TODO: check if these are splitting the data properly -- should be based on JNB tests
        training_batch = (
            jnp.array(self.states[indexes[val_size:]]),
            jnp.array(self.actions[indexes[val_size:]]),
            jnp.array(self.next_states[indexes[val_size:]]),
            jnp.array(self.rewards[indexes[val_size:]]),
            jnp.array(self.not_dones[indexes[val_size:]])
        )

        validation_batch = (
            jnp.array(self.states[indexes[:val_size]]),
            jnp.array(self.actions[indexes[:val_size]]),
            jnp.array(self.next_states[indexes[:val_size]]),
            jnp.array(self.rewards[indexes[:val_size]]),
            jnp.array(self.not_dones[indexes[:val_size]])
        )

        return training_batch, validation_batch


class OfflineReplayJAX:
    def __init__(self, env: Env, device: str, custom_filepath: Union[str, None] = None) -> None:
        import d4rl
        self.env = env
        self.device = device

        if custom_filepath:
            with open(custom_filepath, 'r') as f:
                self.dataset = json.load(f)
                self.states = np.array(self.dataset['observations'])
                self.actions = np.array(self.dataset['actions'])
                self.next_states = np.array(self.dataset['next_observations'])
                self.rewards = np.array(self.dataset['rewards']).reshape(-1, 1)
                self.not_dones = (1 - np.array(self.dataset['terminals'])).reshape(-1, 1)
        else:
            self.dataset = d4rl.qlearning_dataset(env)
            self.states = self.dataset['observations']
            self.actions = self.dataset['actions']
            self.next_states = self.dataset['next_observations']
            self.rewards = self.dataset['rewards'].reshape(-1, 1)
            self.not_dones = (1 - self.dataset['terminals']).reshape(-1, 1)

        self.size = self.rewards.shape[0]

        # The current d4rl datasets are very annoying. For CR, there is no early-termination condition (unlike WW and
        # hopper). Therefore, the below is ONLY FOR CR!~!!!!!@
        self.traj_indices = []

        b_idx = 0
        e_idx = 999

        while e_idx <= self.rewards.shape[0]:
            self.traj_indices.append([b_idx, e_idx])
            b_idx += 1000
            e_idx += 1000

    def random_split(
            self, val_size: int, batch_size: int
    ):  # noqa
        # batch_idxs = np.random.permutation(self.size)[:batch_size]
        batch_idxs = np.random.permutation(
            np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)]
        )[:batch_size]

        training_batch = (
            jnp.array(self.states[batch_idxs[val_size:]]),
            jnp.array(self.actions[batch_idxs[val_size:]]),
            jnp.array(self.next_states[batch_idxs[val_size:]]),
            jnp.array(self.rewards[batch_idxs[val_size:]]),
            jnp.array(self.not_dones[batch_idxs[val_size:]])
        )

        validation_batch = (
            jnp.array(self.states[batch_idxs[:val_size]]),
            jnp.array(self.actions[batch_idxs[:val_size]]),
            jnp.array(self.next_states[batch_idxs[:val_size]]),
            jnp.array(self.rewards[batch_idxs[:val_size]]),
            jnp.array(self.not_dones[batch_idxs[:val_size]])
        )

        return training_batch, validation_batch

    def sample(
            self, batch_size: int, rl: bool = False
    ):
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        # print(f'INDS!: {ind}')
        # ind = np.random.randint(0, self.size, size=batch_size)[self.not_dones == 1]

        return (
            jnp.array(self.states[ind]),
            jnp.array(self.actions[ind]),
            jnp.array(self.next_states[ind]),
            jnp.array(self.rewards[ind]),
            jnp.array(self.not_dones[ind])
        )

    def random_split_transformer(
            self, val_size: int, batch_size: int, h: int, scaler: StandardScaler
    ) -> Tuple[Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]]:  # noqa
        input_batch_s = []
        input_batch_a = []
        target_batch_s = []
        target_batch_r = []
        ts = []

        # TODO: the below routine only works for trajectories that have no early termination conditions.
        # Perhaps we need to also pre-compute and store some information on each trajectories length?
        for i in range(batch_size):
            traj_index = self.traj_indices[np.random.choice(range(len(self.traj_indices)))]
            traj_end = np.random.choice(np.arange(1, 999))
            traj_beginning = traj_end - h

            traj_index_end = traj_index[1] - (999 - traj_end)

            # We will zero-pad (left) trajectories that are before the horizon length
            if traj_beginning < 0:
                traj_beginning = 0

                num_pads = h - traj_end

                traj_index_beginning = traj_index_end - (h - num_pads)

                # [B, object_dim]
                s = torch.from_numpy(self.states[traj_index_beginning: traj_index_end])
                a = torch.from_numpy(self.actions[traj_index_beginning: traj_index_end])

                # [B, |s| + |a|] --> scale and split out
                sa = torch.cat([s, a], dim=-1)
                sa = scaler.transform(sa.to(self.device))
                s = sa[:, :self.states.shape[-1]]
                a = sa[:, self.states.shape[-1]:]

                s = torch.cat([torch.zeros(num_pads, self.states.shape[-1]).to(self.device), s], dim=0)
                a = torch.cat([torch.zeros(num_pads, self.actions.shape[-1]).to(self.device), a], dim=0)

                t = torch.arange(start=0, end=h, step=1)

            else:
                traj_index_beginning = traj_index_end - h

                # [B, object_dim]
                s = torch.from_numpy(self.states[traj_index_beginning: traj_index_end])
                a = torch.from_numpy(self.actions[traj_index_beginning: traj_index_end])

                # [B, |s| + |a|] -->
                sa = torch.cat([s, a], dim=-1)
                sa = scaler.transform(sa.to(self.device))
                s = sa[:, :self.states.shape[-1]]
                a = sa[:, self.states.shape[-1]:]

                t = torch.arange(start=traj_beginning, end=traj_end, step=1)

            input_batch_s.append(s.unsqueeze(0))
            input_batch_a.append(a.unsqueeze(0))

            target_batch_s.append(
                torch.from_numpy(
                    self.states[traj_index_end] - self.states[traj_index_end - 1])
            )

            target_batch_r.append(
                torch.from_numpy(self.rewards[traj_index_end - 1])
            )

            ts.append(t)

        input_batch_s = torch.vstack(input_batch_s).to(self.device)
        input_batch_a = torch.vstack(input_batch_a).to(self.device)
        target_batch_s = torch.vstack(target_batch_s).to(self.device)
        target_batch_r = torch.vstack(target_batch_r).to(self.device)
        ts = torch.vstack(ts).to(self.device)

        train_batch = (
            input_batch_s[val_size:],
            input_batch_a[val_size:],
            target_batch_s[val_size:],
            target_batch_r[val_size:],
            ts[val_size:]
        )

        val_batch = (
            input_batch_s[:val_size],
            input_batch_a[:val_size],
            target_batch_s[:val_size],
            target_batch_r[:val_size],
            ts[:val_size]
        )

        return train_batch, val_batch