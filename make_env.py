import random
from typing import Any, Optional

import numpy as np
import gymnasium as gym
import envpool
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from discretization import DiscretizationWrapper
from config import Config


class VecAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a SB3 VecEnv.
    """

    def __init__(self, venv, disable_terminal_obs: bool = False):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)

        self.disable_terminal_obs = disable_terminal_obs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        return self.venv.reset()[0]

    def seed(self, seed: Optional[int] = None) -> None:
        pass

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        dones = terms + truncs
        infos = []
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], np.ndarray)
                }
            )
            if dones[i]:
                if not self.disable_terminal_obs:
                    infos[i]["terminal_observation"] = np.copy(obs[i])
                    infos[i]["TimeLimit.truncated"] = truncs[i]
                obs[i] = self.venv.reset(np.array([i]))[0]
        return obs, rewards, dones, infos


def make_env(config: Config, training: bool = True, seed: Optional[int] = None):
    n_envs = config.n_envs if training else config.n_eval_envs
    env_id = config.env

    if config.minigrid:
        import minigrid
        import minigrid.wrappers
        from minigrid.wrappers import ImgObsWrapper

        def make_minigrid():
            env = gym.make(env_id, render_mode="rgb_array")

            class NoStepForTurning(gym.Wrapper):
                def __init__(self, env: gym.Env):
                    super().__init__(env)
                    self._forward_count = 0

                def reset(self, *args, **kwargs):
                    self._forward_count = 0
                    return super().reset(*args, **kwargs)

                def step(self, action):
                    is_turn = action <= 1

                    self.unwrapped.step_count -= is_turn
                    step_result = self.env.step(action)

                    if is_turn:
                        step_result = self.env.step(action=2)

                    return step_result

            if "Empty" in env_id:
                env = NoStepForTurning(env)

            Wrapper: type[gym.Wrapper]
            if "minigrid_wrapper" in config.env_kwargs:
                Wrapper = getattr(
                    minigrid.wrappers, config.env_kwargs["minigrid_wrapper"]
                )
                env = Wrapper(env)

            env = ImgObsWrapper(env)
            return env

        env = make_vec_env(
            make_minigrid,
            n_envs=n_envs,
            seed=seed,
        )

    elif env_id in envpool.list_all_envs():
        kwargs = (
            config.env_kwargs
            if training
            else (
                config.eval_env_kwargs
                if config.eval_env_kwargs is not None
                else config.env_kwargs
            )
        )
        if seed is not None:
            kwargs["seed"] = seed
        elif config.seed is not None:
            kwargs["seed"] = config.seed
        else:
            kwargs["seed"] = random.randint(-(2**31), 2**31 - 1)
        if config.procgen:
            kwargs["num_levels"] = (
                config.procgen_train_num_levels
                if training
                else config.procgen_eval_num_levels
            )

        env = envpool.make(env_id, env_type="gymnasium", num_envs=n_envs, **kwargs)
        env.spec.id = env_id
        env = VecAdapter(env, disable_terminal_obs=config.procgen)
    else:
        raise NotImplementedError()

    if config.action_discretization:
        env = DiscretizationWrapper(env, config.discretization_num_atomics)

    env = VecMonitor(env)

    if config.norm_reward or config.norm_obs:
        env = VecNormalize(
            env,
            training=training,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            gamma=(
                config.ppo_config.gamma
                if config.algo == "PPO"
                else config.trpo_config.gamma
            ),
        )

    return env
