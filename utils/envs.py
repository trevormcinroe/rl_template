import numpy as np
from gym import Env
from typing import Union, Tuple, Any, Dict


class DMCWrapper:
    def __init__(self, env: Env, action_repeat: int):
        self.env = env
        self.action_repeat = action_repeat

    def reset(self) -> np.array:
        timestep = self.env.reset()
        obs = self.extract_obs(timestep)
        return obs

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        reward = 0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward
        return self.extract_obs(timestep), reward, timestep.step_type == 2, {}

    def extract_obs(self, timestep: Dict[str: Any]) -> np.array:
        return np.concatenate(list(timestep.observation.values()))

    # @property
    def sample_action(self) -> np.array:
        return np.random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)
