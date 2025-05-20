from typing import Literal, Generator, NamedTuple

import numpy as np

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import VecNormalize


class PopArtLayer(nn.Module):
    mu: th.Tensor
    sigma: th.Tensor

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float = 3.0e-4,
        init: Literal["kaiming", "scaled_normalised", "ortho"] = "kaiming",
        scale: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.beta = beta

        self.weight = nn.Parameter(th.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(th.Tensor(output_dim))

        mu = th.zeros(output_dim, dtype=th.float32)
        sigma = th.zeros(output_dim, dtype=th.float32)
        nu = th.zeros(output_dim, dtype=th.float32)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("nu", nu)

        match init:
            case "kaiming":
                nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            case "scaled_normalised":
                th.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
                self.weight.data *= scale / self.weight.norm(dim=1, p=2, keepdim=True)
                self.bias.data.zero_()
            case "ortho":
                nn.init.orthogonal_(self.weight, gain=gain)  # type: ignore
                self.bias.data.zero_()

        self.output_dim = output_dim

    def forward(self, x: th.Tensor):
        normalised_y = x.mm(self.weight.t())
        normalised_y = normalised_y + self.bias.unsqueeze(0).expand_as(normalised_y)
        return normalised_y

    @th.no_grad()
    def denormalise(self, normalised_y: th.Tensor):
        return normalised_y * self.sigma + self.mu

    def denormalise_numpy(self, normalised_y: np.ndarray) -> np.ndarray:
        return (
            self.denormalise(th.from_numpy(normalised_y).to(self.nu.device))
            .cpu()
            .numpy()
        )

    def normalise(self, unnormalised_y: np.ndarray) -> np.ndarray:
        return (
            self.normalise_tensor(th.from_numpy(unnormalised_y).to(self.nu.device))
            .cpu()
            .numpy()
        )

    def normalise_tensor(self, unnormalised_y: th.Tensor):
        return (unnormalised_y - self.mu) / self.sigma

    def update_stats_and_params(self, target: np.ndarray) -> np.ndarray:
        return (
            self.update_and_normalise_tensor(th.from_numpy(target).to(self.nu.device))
            .cpu()
            .numpy()
        )

    def update_and_normalise(self, target: np.ndarray) -> np.ndarray:
        self.update_stats_and_params(target)
        return self.normalise(target)

    @th.no_grad()
    def update_and_normalise_tensor(
        self, target: th.Tensor, beta: float | None = None
    ) -> th.Tensor:
        beta = beta or self.beta
        old_mu, old_nu, old_sigma = self.mu, self.nu, self.sigma

        mu = target.mean()
        nu = (target**2).mean()

        new_mu = (1 - beta) * old_mu + beta * mu
        new_nu = (1 - beta) * old_nu + beta * nu
        new_sigma = th.clip(th.sqrt(new_nu - new_mu**2 + 1e-8), 1e-4, 1e6)

        self.weight.data = self.weight.data * old_sigma / new_sigma
        self.bias.data = (old_sigma * self.bias.data + old_mu - new_mu) / new_sigma

        self.mu = new_mu
        self.nu = new_nu
        self.sigma = new_sigma

        return (target - new_mu) / new_sigma
