from dataclasses import dataclass, field
from functools import partial
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    PyTorchObs,
)
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.torch_layers import (
    NatureCNN,
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from gymnasium import spaces

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from rollout_buffer_with_entropy import RolloutBufferWithEntropy
from policy import EAPOActorCritic
from config import Config
from impala_cnn import ImpalaCNN
from minigrid_features_extractor import (
    MinigridFeaturesExtractor,
)
from popart import PopArtLayer


class EAPO(PPO):
    policy: EAPOActorCritic
    rollout_buffer: RolloutBufferWithEntropy

    def __init__(
        self,
        env: GymEnv,
        device: th.device | str,
        config: Config = Config(),
        verbose: int = 0,
        seed: int | None = None,
        _init_setup_model: bool = True,
        policy=EAPOActorCritic,
    ):
        self.config = config
        ppo_config = config.ppo_config
        policy_kwargs = {"config": config, **ppo_config.policy_kwargs}

        self.use_entropy_advantage = config.eapo_config.use_entropy_advantage
        self.augmented_reward = config.eapo_config.augmented_reward
        self.tau = config.eapo_config.tau
        self.c2 = config.eapo_config.c2
        self.tau_on_entropy = config.eapo_config.tau_on_entropy

        if config.procgen:
            policy_kwargs["features_extractor_class"] = ImpalaCNN
            policy_kwargs["net_arch"] = []
        elif config.minigrid:
            policy_kwargs["features_extractor_class"] = MinigridFeaturesExtractor
            policy_kwargs["net_arch"] = []
        elif config.env == "MontezumaRevenge-v5":
            policy_kwargs["features_extractor_class"] = NatureCNN
            policy_kwargs["net_arch"] = []

        super().__init__(
            policy=EAPOActorCritic,
            env=env,
            learning_rate=ppo_config.learning_rate,
            n_steps=ppo_config.n_steps,
            batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            clip_range=ppo_config.clip_range,
            clip_range_vf=ppo_config.clip_range_vf,
            normalize_advantage=ppo_config.normalize_advantage,
            ent_coef=ppo_config.ent_coef,
            vf_coef=ppo_config.vf_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=ppo_config.tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()

        self.e_gamma = (
            self.config.eapo_config.e_gamma
            if self.config.eapo_config.e_gamma is not None
            else self.gamma
        )
        self.e_lambda = (
            self.config.eapo_config.e_lambda
            if self.config.eapo_config.e_lambda is not None
            else self.gae_lambda
        )

        self.rollout_buffer = RolloutBufferWithEntropy(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            self.gamma,
            self.gae_lambda,
            self.n_envs,
            self.e_gamma,
            self.e_lambda,
        )

        self.policy.tau = self.tau

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBufferWithEntropy,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (
                    actions,
                    values,
                    log_probs,
                    predicted_total_entropies,
                    entropies,
                ) = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            # entropies = -log_probs.cpu().clone().numpy()
            entropies = entropies.cpu().numpy()
            if self.tau_on_entropy:
                entropies *= self.tau

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            values, predicted_total_entropies = self.policy.maybe_popart_denormalise(
                values, predicted_total_entropies
            )

            if self.config.handle_timeout:
                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(
                            infos[idx]["terminal_observation"]
                        )[0]
                        with th.no_grad():
                            terminal_value, terminal_entropy_preds = self.policy.predict_values(terminal_obs)  # type: ignore[arg-type]

                        terminal_value, terminal_entropy_preds = (
                            self.policy.maybe_popart_denormalise(
                                terminal_value, terminal_entropy_preds
                            )
                        )

                        rewards[idx] += self.gamma * terminal_value.item()
                        entropy_correction = self.e_gamma * terminal_entropy_preds  # type: ignore
                        entropies[idx] += entropy_correction.item()

            if self.augmented_reward:
                rewards = rewards + entropies * self.tau

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                entropies,
                predicted_total_entropies,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values, predicted_total_entropies = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            values, predicted_total_entropies = self.policy.maybe_popart_denormalise(
                values, predicted_total_entropies
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_entropy(predicted_total_entropies, dones)

        (
            rollout_buffer.returns,
            rollout_buffer.total_entropies,
        ) = self.policy.maybe_update_and_normalise_popart(
            rollout_buffer.returns, rollout_buffer.total_entropies
        )

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        entropy_pred_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, entropy_preds = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                entropy_preds = entropy_preds.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                entropy_advantages = rollout_data.entropy_advantages

                if not self.tau_on_entropy:
                    entropy_advantages *= self.tau
                if self.use_entropy_advantage and not self.augmented_reward:
                    advantages = advantages + entropy_advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy critic loss
                if not self.augmented_reward and self.use_entropy_advantage:
                    entropy_targets = rollout_data.total_entropies
                    entropy_preds = entropy_preds.squeeze()
                    entropy_prediction_loss = F.mse_loss(entropy_preds, entropy_targets)
                    entropy_pred_losses.append(entropy_prediction_loss.item())
                    value_loss = value_loss + self.c2 * entropy_prediction_loss

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        if isinstance(self.policy.value_net, PopArtLayer):
            normalised_values = self.policy.value_net.normalise(
                self.rollout_buffer.values.flatten()
            )
            explained_var = explained_variance(
                normalised_values, self.rollout_buffer.returns.flatten()
            )
        else:
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )

        if isinstance(self.policy.entropy_net, PopArtLayer):
            normalised_ent_preds = self.policy.entropy_net.normalise(
                self.rollout_buffer.predicted_total_entropies.flatten()
            )
            entropy_explained_var = explained_variance(
                normalised_ent_preds,
                self.rollout_buffer.total_entropies.flatten(),
            )
        else:
            entropy_explained_var = explained_variance(
                self.rollout_buffer.predicted_total_entropies.flatten(),
                self.rollout_buffer.total_entropies.flatten(),
            )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_explained_variance", entropy_explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self.logger.record("train/entropy_critic_loss", np.mean(entropy_pred_losses))
        if isinstance(self.policy.value_net, PopArtLayer):
            self.logger.record("pop_art/value_mu", self.policy.value_net.mu.item())
            self.logger.record(
                "pop_art/value_sigma", self.policy.value_net.sigma.item()
            )
        if isinstance(self.policy.entropy_net, PopArtLayer):
            self.logger.record("pop_art/entropy_mu", self.policy.entropy_net.mu.item())
            self.logger.record(
                "pop_art/entropy_sigma", self.policy.entropy_net.sigma.item()
            )

    def learn(
        self: "EAPO",
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "EAPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "EAPO":
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
