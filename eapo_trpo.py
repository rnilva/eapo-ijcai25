import copy
from functools import partial
from typing import Any, Callable

import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium.spaces as spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.trpo import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver

from policy import EAPOActorCritic
from rollout_buffer_with_entropy import (
    RolloutBufferWithEntropy,
    RolloutBufferSamplesWithEntropy,
)
from config import Config, EAPOConfig
from impala_cnn import ImpalaCNN
from popart import PopArtLayer


class EAPO_TRPO(TRPO):
    policy: EAPOActorCritic
    rollout_buffer: RolloutBufferWithEntropy

    def __init__(
        self,
        env: VecEnv,
        config: Config,
        eapo_config: EAPOConfig,
        learning_rate: float | Callable[[float], float] = 0.001,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        trpo_reverse_kl: bool = True,
        sub_sampling_factor: int = 1,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            EAPOActorCritic,
            env,
            learning_rate,
            n_steps,
            batch_size,
            gamma,
            cg_max_steps,
            cg_damping,
            line_search_shrinking_factor,
            line_search_max_iter,
            n_critic_updates,
            gae_lambda,
            use_sde,
            sde_sample_freq,
            None,  # rollout_buffer_class,
            None,  # rollout_buffer_kwargs,
            normalize_advantage,
            target_kl,
            sub_sampling_factor,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            False,  # _init_setup_model,
        )

        self.use_entropy_advantage = eapo_config.use_entropy_advantage
        self.tau = eapo_config.tau
        self.e_gamma = eapo_config.e_gamma or self.gamma
        self.e_lambda = eapo_config.e_lambda or self.gae_lambda
        self.c2 = eapo_config.c2
        self.tau_on_entropy = eapo_config.tau_on_entropy
        self.handle_timeout = config.handle_timeout
        self.augmented_reward = eapo_config.augmented_reward
        self.kl_reversed = trpo_reverse_kl

        self.policy_kwargs["config"] = config
        if config.procgen:
            self.policy_kwargs["features_extractor_class"] = ImpalaCNN
            self.policy_kwargs["net_arch"] = []

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

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

            if self.handle_timeout:
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

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []
        entropy_pred_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamplesWithEntropy(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # returns, not used here,
                    None,  # type: ignore[arg-type]  # entropy returns, not used here,
                    rollout_data.entropy_advantages[:: self.sub_sampling_factor],
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(
                    self.policy.get_distribution(rollout_data.observations)
                )

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.use_entropy_advantage and not self.augmented_reward:
                entropy_advantages = rollout_data.entropy_advantages
                if not self.tau_on_entropy:
                    entropy_advantages *= self.tau
                advantages = advantages + entropy_advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    rollout_data.advantages.std() + 1e-8
                )

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()

            # KL divergence
            if self.kl_reversed:
                kl_div = kl_divergence(distribution, old_distribution).mean()
            else:
                kl_div = kl_divergence(old_distribution, distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = (
                self._compute_actor_grad(kl_div, policy_objective)
            )

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(
                self.hessian_vector_product, actor_params, grad_kl
            )

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction,
                hessian_vector_product_fn(search_direction, retain_graph=False),
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)  # type: ignore[assignment, arg-type]

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    # Applying the scaled step direction
                    for param, original_param, shape in zip(
                        actor_params, original_actor_params, grad_shape
                    ):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(
                                shape
                            )
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(
                        rollout_data.observations
                    )
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()

                    # New KL-divergence
                    if self.kl_reversed:
                        kl_div = kl_divergence(distribution, old_distribution).mean()
                    else:
                        kl_div = kl_divergence(old_distribution, distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (kl_div < self.target_kl) and (
                        new_policy_objective > policy_objective
                    ):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(
                        actor_params, original_actor_params
                    ):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred, entropy_pred = self.policy.predict_values(
                    rollout_data.observations
                )
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                if self.use_entropy_advantage and not self.augmented_reward:
                    entropy_prediction_loss = F.mse_loss(
                        rollout_data.total_entropies, entropy_pred.flatten()
                    )
                    entropy_pred_losses.append(entropy_prediction_loss.item())
                    value_loss = value_loss + self.c2 * entropy_prediction_loss

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1

        normalised_values = self.policy.value_net.normalise(
            self.rollout_buffer.values.flatten()
        )
        explained_var = explained_variance(
            normalised_values, self.rollout_buffer.returns.flatten()
        )
        normalised_ent_preds = self.policy.entropy_net.normalise(
            self.rollout_buffer.predicted_total_entropies.flatten()
        )
        entropy_explained_var = explained_variance(
            normalised_ent_preds,
            self.rollout_buffer.total_entropies.flatten(),
        )

        # Entropy loss just for logging
        with th.no_grad():
            entropy = distribution.entropy()
            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_critic_loss", np.mean(entropy_pred_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_explained_variance", entropy_explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

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

        self.logger.record("train/entropy_loss", entropy_loss.item())

    def learn(
        self: "EAPO_TRPO",
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "EAPO_TRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "EAPO_TRPO":
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
