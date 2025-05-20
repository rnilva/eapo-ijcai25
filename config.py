from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = 0
    tensorboard_log: str | None = None
    policy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TRPOConfig:
    learning_rate: float = 1e-3
    n_steps: int = 2048
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = 0
    policy_kwargs: dict[str, Any] = field(default_factory=dict)

    cg_max_steps: int = 15
    cg_damping: float = 0.1
    line_search_shrinking_factor: float = 0.8
    line_search_max_iter: int = 10
    n_critic_updates: int = 10
    sub_sampling_factor: int = 1
    target_kl: float = 0.01
    trpo_reverse_kl: bool = True


@dataclass
class EAPOConfig:
    use_entropy_advantage: bool = True
    augmented_reward: bool = False
    tau: float = 0.01
    c2: float = 0.5
    e_gamma: float | None = None
    e_lambda: float | None = None
    ea_coef: float = 1.0
    tau_on_entropy: bool = False


@dataclass
class Config:
    env: str = ""
    algo: Literal["PPO", "TRPO"] = "PPO"
    total_timesteps: int = int(1e7)

    device: str = "auto"
    seed: int | None = None
    verbose: int = 1

    ppo_config: PPOConfig = PPOConfig()
    trpo_config: TRPOConfig = TRPOConfig()
    eapo_config: EAPOConfig = EAPOConfig()

    n_eval_episodes: int = 100
    n_eval_envs: int = 32
    eval_freq: int = 2000
    eval_verbose: int = 1

    handle_timeout: bool = True
    pop_art: bool = True
    pop_art_beta: float = 3.0e-4

    n_envs: int = 32
    norm_obs: bool = False
    norm_reward: bool = False
    action_discretization: bool = False
    discretization_num_atomics: int = 7

    procgen: bool = False
    minigrid: bool = False
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: dict[str, Any] | None = None
    procgen_train_num_levels: int = 200
    procgen_eval_num_levels: int = 0
