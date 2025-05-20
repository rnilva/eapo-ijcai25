from typing import NamedTuple, Generator, Literal

import numpy as np
import torch as th

from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class RolloutBufferSamplesWithEntropy(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    total_entropies: th.Tensor
    entropy_advantages: th.Tensor


class RolloutBufferWithEntropy(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str,
        gamma: float,
        gae_lambda: float,
        n_envs: int,
        e_gamma: float | None,
        e_lambda: float | None,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

        self.e_lambda = e_lambda or gae_lambda
        self.e_gamma = e_gamma or gamma

    def compute_entropy(self, last_entropy: th.Tensor, dones: np.ndarray) -> None:
        assert self.episode_starts is not None
        assert self.log_probs is not None

        last_total_entropy: np.ndarray | Literal[0] = 0
        last_total_entropy = 0

        for step in reversed(range(self.buffer_size)):
            next_non_terminal: np.ndarray
            next_entropy: np.ndarray
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_entropy = last_entropy.clone().cpu().numpy().flatten()
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_entropy = self.predicted_total_entropies[step + 1]
            delta = (
                self.entropies[step]
                + self.e_gamma * next_entropy * next_non_terminal
                - self.predicted_total_entropies[step]
            )

            last_total_entropy = (
                delta
                + self.e_gamma * self.e_lambda * next_non_terminal * last_total_entropy
            )
            self.entropy_advantage[step] = last_total_entropy

        self.total_entropies = self.entropy_advantage + self.predicted_total_entropies

    def reset(self) -> None:
        self.entropies = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.total_entropies = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.predicted_total_entropies = np.zeros_like(self.total_entropies)
        self.entropy_advantage = np.zeros_like(self.total_entropies)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        entropies: np.ndarray,
        predicted_total_entropy: th.Tensor | None,
    ) -> None:
        self.entropies[self.pos] = entropies

        if predicted_total_entropy is not None:
            self.predicted_total_entropies[self.pos] = (
                predicted_total_entropy.clone().cpu().numpy().flatten()
            )

        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: int | None = None
    ) -> Generator[RolloutBufferSamplesWithEntropy, None, None]:
        if not self.generator_ready:
            self.predicted_total_entropies = self.swap_and_flatten(
                self.predicted_total_entropies
            )
            self.total_entropies = self.swap_and_flatten(self.total_entropies)
            self.entropy_advantage = self.swap_and_flatten(self.entropy_advantage)
        return super().get(batch_size)  # type: ignore

    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> RolloutBufferSamplesWithEntropy:
        sample = super()._get_samples(batch_inds, env)
        return RolloutBufferSamplesWithEntropy(
            *sample,
            self.to_torch(self.total_entropies[batch_inds].flatten()),
            self.to_torch(self.entropy_advantage[batch_inds].flatten()),
        )
