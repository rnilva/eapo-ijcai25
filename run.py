import random
import yaml
from argparse import ArgumentParser
from dataclasses import asdict

from config import Config, EAPOConfig, PPOConfig, TRPOConfig
from make_env import make_env
from evaluate_policy import EvalCallback
from eapo import EAPO
from eapo_trpo import EAPO_TRPO

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_dict = yaml.load(f, yaml.FullLoader)
        yaml_dict["ppo_config"] = PPOConfig(**yaml_dict["ppo_config"])
        yaml_dict["eapo_config"] = EAPOConfig(**yaml_dict["eapo_config"])
        yaml_dict["trpo_config"] = TRPOConfig(**yaml_dict["trpo_config"])
        config = Config(**yaml_dict)

    if config.seed is None:
        config.seed = random.randint(0, 2**31 - 1)

    env = make_env(config, training=True, seed=config.seed + 17)
    eval_env = make_env(config, training=False, seed=config.seed + 27)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config.n_eval_episodes,
        eval_freq=config.eval_freq,
        deterministic=False,
        verbose=config.eval_verbose,
    )

    if config.algo == "PPO":
        model = EAPO(
            env, config.device, config, verbose=config.verbose, seed=config.seed
        )
    elif config.algo == "TRPO":
        model = EAPO_TRPO(
            env,
            config,
            config.eapo_config,
            verbose=config.verbose,
            seed=config.seed,
            **asdict(config.trpo_config)
        )
    else:
        raise NotImplementedError(config.algo)

    model.learn(config.total_timesteps, callback=eval_callback)
