# Overrides the SB3's evaluate_policy to record episodic trajectory entropy.
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, Protocol

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium import Env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback as SB3EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)


class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...] | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action, log probability, and the next hidden state
            (used in recurrent policies)
        """
        ...

    def predict_values(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]: ...


def evaluate_policy(
    model: PolicyPredictor,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    warn: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_entropies = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_entropies = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, log_probs, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_entropies += -log_probs
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                    episode_entropies.append(current_entropies[i])

                    current_rewards[i] = 0
                    current_entropies[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    # mean_reward = np.mean(episode_rewards)
    # std_reward = np.std(episode_rewards)
    # if reward_threshold is not None:
    #     assert mean_reward > reward_threshold, (
    #         "Mean reward below threshold: "
    #         f"{mean_reward:.2f} < {reward_threshold:.2f}"
    #     )
    # if return_episode_rewards:
    #     return episode_rewards, episode_entropies, episode_lengths
    # return mean_reward, std_reward
    return episode_rewards, episode_entropies, episode_lengths


class EvalCallback(SB3EvalCallback):
    def __init__(
        self,
        eval_env: Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )

        self.log_key = "eval"

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_entropies, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            mean_traj_entropy = np.mean(episode_entropies)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.log_key}/mean_reward", float(mean_reward))
            self.logger.record(
                f"{self.log_key}/mean_traj_entropy", float(mean_traj_entropy)
            )
            self.logger.record(f"{self.log_key}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.log_key}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    entropies: np.ndarray
    returns: np.ndarray = np.empty(0)
    entropy_returns: np.ndarray = np.empty(0)
    lengths: np.ndarray = np.empty(0)
    advantages: np.ndarray = np.empty(0)
    entropy_advantages: np.ndarray = np.empty(0)
    total_advantages: np.ndarray = np.empty(0)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.returns[idx],
            self.entropy_returns[idx],
        )

    def __len__(self):
        return len(self.observations)


def collect_trajectories(policy: PolicyPredictor, env: VecEnv, n_traj: int):
    N = env.num_envs
    LMAX = env.unwrapped.spec.config.max_episode_steps
    I = np.arange(N)
    L = np.arange(LMAX)[None, :]

    trajectories: list[Trajectory] = []

    current_observations = np.zeros(
        (N, LMAX, *env.observation_space.shape), dtype=env.observation_space.dtype
    )
    current_actions = np.zeros(
        (N, LMAX, *env.action_space.shape), dtype=env.action_space.dtype
    )
    current_rewards = np.zeros((N, LMAX))
    current_entropies = np.zeros_like(current_rewards)

    n = 0
    l = np.zeros(N, dtype=np.int32)

    obs = env.reset()

    # pbar = tqdm(total=n_traj, desc="Collecting Trajectories...")

    while n < n_traj:
        a, log_prob, _ = policy.predict(obs)
        next_obs, r, done, infos = env.step(a)
        entropy = -log_prob

        current_observations[I, l, ...] = obs
        current_actions[I, l, ...] = a
        current_rewards[I, l] = r
        current_entropies[I, l] = entropy

        l += 1

        obs = next_obs

        if done.any():

            def take_with_lengths(
                a: np.ndarray, i: np.ndarray, m: np.ndarray, l: np.ndarray
            ) -> list[np.ndarray]:
                return np.split(a[i][m], np.cumsum(l)[:-1])

            d = np.where(done)[0]
            epi_l = l[d]
            mask = L < epi_l[:, None]
            done_obss = take_with_lengths(current_observations, d, mask, epi_l)
            done_actions = take_with_lengths(current_actions, d, mask, epi_l)
            done_rewards = take_with_lengths(current_rewards, d, mask, epi_l)
            done_entropies = take_with_lengths(current_entropies, d, mask, epi_l)
            for d_o, d_a, d_r, d_e in zip(
                done_obss, done_actions, done_rewards, done_entropies
            ):
                traj = Trajectory(d_o, d_a, d_r, d_e)
                trajectories.append(traj)

            num_dones = np.sum(done)
            n += num_dones
            l[done] = 0

            # pbar.update(num_dones)

    # pbar.close()

    return trajectories


def calculate_returns(
    traj: Trajectory,
    gamma: float,
    e_gamma: float,
    gae_lambda: float,
    e_lambda: float,
    tau: float,
    policy: PolicyPredictor,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l = len(traj.observations)

    with th.no_grad():
        obs_t = th.as_tensor(traj.observations, device=device)
        values, e_values = policy.predict_values(obs_t)
        values = values.cpu().numpy().flatten()
        e_values = e_values.cpu().numpy().flatten()

    A = np.zeros(l)
    EA = np.zeros_like(A)
    L = np.zeros_like(A)
    v_gae_total = 0.0
    e_gae_total = 0.0
    for i in reversed(range(l)):
        r, e = traj.rewards[i], traj.entropies[i]
        if i == l - 1:
            A[i] = r
            EA[i] = e
            L[i] = 1
        else:
            v_delta = r + gamma * values[i + 1]
            e_delta = e + e_gamma * e_values[i + 1]

            v_gae_total = v_delta + gamma * gae_lambda * v_gae_total
            e_gae_total = e_delta + e_gamma * e_lambda * e_gae_total

            A[i] = v_gae_total
            EA[i] = e_gae_total
            L[i] = 1 + L[i + 1]

    G = A + values
    H = EA + e_values

    traj.returns = G
    traj.entropy_returns = H
    traj.lengths = L
    traj.advantages = A
    traj.entropy_advantages = EA
    traj.total_advantages = A + tau * EA
    return G, H, L
