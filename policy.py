from functools import partial
from typing import Any


import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
)

from config import Config
from popart import PopArtLayer
from impala_cnn import ImpalaCNN


class EAPOActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        config: Config,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        self.config = config

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        elif isinstance(
            self.action_dist,
            (
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        if self.ortho_init:
            init = "ortho"
        elif self.features_extractor_class == ImpalaCNN:
            init = "scaled_normalised"
            assert isinstance(self.action_space, spaces.Discrete)
            self.action_net = ImpalaCNN._linear_normalised_init(
                self.mlp_extractor.latent_dim_pi, int(self.action_space.n), 0.1
            )
        else:
            init = "kaiming"

        if self.config.pop_art:
            self.value_net = PopArtLayer(
                self.mlp_extractor.latent_dim_vf,
                1,
                self.config.pop_art_beta,
                init=init,
                scale=0.1,
                gain=1,
            )
        else:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if not self.config.pop_art:
            self.entropy_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        else:
            self.entropy_net = PopArtLayer(
                self.mlp_extractor.latent_dim_vf,
                1,
                self.config.pop_art_beta,
                init=init,
                scale=0.1,
                gain=1,
            )

        self.log_std_net = None
        if isinstance(self.action_dist, DiagGaussianDistribution):
            # State and action dependent std net
            self.log_std_net = nn.Linear(
                self.mlp_extractor.latent_dim_pi, self.action_dist.action_dim
            )
            nn.init.normal_(self.log_std_net.weight, std=0.01)
            with th.no_grad():
                self.log_std_net.bias.zero_()
                self.log_std_net.bias += np.log(np.exp(np.exp(self.log_std_init)) - 1)
            delattr(self, "log_std")

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                # self.value_net: 1,
            }

            if not self.config.pop_art:
                module_gains[self.value_net] = 1
                module_gains[self.entropy_net] = 1

            if self.features_extractor_class == NatureCNN:
                if not self.share_features_extractor:
                    # Note(antonin): this is to keep SB3 results
                    # consistent, see GH#1148
                    module_gains[self.pi_features_extractor] = np.sqrt(2)
                    module_gains[self.vf_features_extractor] = np.sqrt(2)
                else:
                    module_gains[self.features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if self.log_std_net is not None:
            self.log_std = th.log(F.softplus(self.log_std_net(latent_pi)))

        return super()._get_action_dist_from_latent(latent_pi)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        entropy_preds = self.entropy_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        entropies = -log_prob
        return actions, values, log_prob, entropy_preds, entropies

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy_preds = self.entropy_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy, entropy_preds

    def predict_values(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        features = BasePolicy.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), self.entropy_net(latent_vf)

    def maybe_update_and_normalise_popart(
        self, returns: np.ndarray, entropy_returns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.value_net, PopArtLayer):
            normalised_returns = self.value_net.update_and_normalise(returns)
        else:
            normalised_returns = returns

        if isinstance(self.entropy_net, PopArtLayer):
            normalised_entropy_returns = self.entropy_net.update_and_normalise(
                entropy_returns
            )
        else:
            normalised_entropy_returns = entropy_returns

        return normalised_returns, normalised_entropy_returns

    def maybe_update_popart(self, returns: np.ndarray, entropy_returns: np.ndarray):
        if isinstance(self.value_net, PopArtLayer):
            self.value_net.update_stats_and_params(returns)
        if isinstance(self.entropy_net, PopArtLayer):
            self.entropy_net.update_stats_and_params(entropy_returns)

    def maybe_popart_denormalise(
        self,
        predicted_returns: th.Tensor,
        predicted_entropy_returns: th.Tensor | None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        if isinstance(self.value_net, PopArtLayer):
            predicted_returns = self.value_net.denormalise(predicted_returns)
        if predicted_entropy_returns is not None and isinstance(
            self.entropy_net, PopArtLayer
        ):
            predicted_entropy_returns = self.entropy_net.denormalise(
                predicted_entropy_returns
            )
        return predicted_returns, predicted_entropy_returns

    def predict(
        self,
        observation: PyTorchObs,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if (
            isinstance(observation, tuple)
            and len(observation) == 2
            and isinstance(observation[1], dict)
        ):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            # actions = self._predict(obs_tensor, deterministic=deterministic)
            distribution = self.get_distribution(obs_tensor)
            actions = distribution.get_actions(deterministic=deterministic)
            log_probs = distribution.log_prob(actions).cpu().numpy()

        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)
            log_probs = log_probs.squeeze(axis=0)

        return actions, log_probs, state  # type: ignore[return-value]
