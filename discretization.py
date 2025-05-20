import numpy as np
import gymnasium.spaces as spaces
from gymnasium import Env, ActionWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn


class DiscretizationWrapper(VecEnvWrapper):
    action_space: spaces.MultiDiscrete

    def __init__(self, venv: VecEnv, K: int):
        assert isinstance(venv.action_space, spaces.Box)
        n = venv.action_space.shape[0]
        action_space = spaces.MultiDiscrete([K] * n)
        super().__init__(venv, action_space=action_space)

        low, high = venv.action_space.low, venv.action_space.high
        self.action_table = np.reshape(
            [np.linspace(low[i], high[i], K) for i in range(n)], [n, K]
        )
        self.n = n

    def action(self, action: np.ndarray):
        continuous_action = self.action_table[np.arange(self.n), action]
        return continuous_action

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        continuous_actions = self.action(self.actions)
        step_results = self.venv.step(continuous_actions)

        return step_results

    def reset(self):
        return self.venv.reset()
