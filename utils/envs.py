import numpy as np


class DMCWrapper:
    def __init__(self, env, action_repeat):
        self.env = env
        self.action_repeat = action_repeat

    def reset(self):
        timestep = self.env.reset()
        obs = self.extract_obs(timestep)
        return obs

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward
        return self.extract_obs(timestep), reward, timestep.step_type == 2, {}

    def extract_obs(self, timestep):
        return np.concatenate(list(timestep.observation.values()))

    # @property
    def sample_action(self):
        return np.random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)