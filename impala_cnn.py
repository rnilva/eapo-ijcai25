from math import sqrt
from typing import List

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium.spaces import Box, Space
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    https://github.com/openai/baselines/blob/master/baselines/common/models.py#L28
    """

    def __init__(
        self,
        observation_space: Space,
        features_dim: int = 256,
        depths: List[int] = [16, 32, 32],
        n_residual_blocks_per_stack: int = 2,
        max_pool: bool = True,
    ):
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False)

        n_input_channels = observation_space.shape[0]
        scale = 1 / sqrt(len(depths))
        stacks: List[nn.Module] = []
        for depth in depths:
            stack = ImpalaCNN.DownsampleStack(
                n_input_channels, depth, n_residual_blocks_per_stack, scale, max_pool
            )
            n_input_channels = depth
            stacks.append(stack)
        stacks.append(nn.Flatten())
        stacks.append(nn.ReLU())
        self.stacks = nn.Sequential(*stacks)

        # Compute shape by doing one forward pass
        with th.no_grad():
            x = th.as_tensor(observation_space.sample()[None]).float()
            y = self.stacks(x)
            n_flatten = y.shape[1]

        self.linear = nn.Sequential(
            ImpalaCNN._linear_normalised_init(n_flatten, features_dim, scale=1.4),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.stacks(observations))

    @staticmethod
    def _conv_layer_normalised_init(
        in_channels: int, out_channels: int, scale: float = 1.0
    ) -> nn.Conv2d:
        l = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        l.weight.data *= scale / l.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
        if l.bias is not None:
            l.bias.data.zero_()
        return l

    @staticmethod
    def _linear_normalised_init(
        in_features: int, out_features: int, scale: float = 1.0
    ):
        l = nn.Linear(in_features, out_features)
        l.weight.data *= scale / l.weight.norm(dim=1, p=2, keepdim=True)
        l.bias.data.zero_()
        return l

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, scale: float = 1.0) -> None:
            super().__init__()
            self.c0 = ImpalaCNN._conv_layer_normalised_init(
                in_channels, in_channels, scale=sqrt(scale)
            )
            self.c1 = ImpalaCNN._conv_layer_normalised_init(
                in_channels, in_channels, scale=sqrt(scale)
            )

        def forward(self, x: th.Tensor) -> th.Tensor:
            x0 = x
            x = th.relu(x)
            x = self.c0(x)
            x = th.relu(x)
            x = self.c1(x)
            return x + x0

    class DownsampleStack(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_residual_blocks: int,
            scale: float,
            max_pool: bool = True,
        ) -> None:
            super().__init__()
            self.input_conv = ImpalaCNN._conv_layer_normalised_init(
                in_channels, out_channels
            )
            self.max_pool = max_pool
            scale /= sqrt(n_residual_blocks)
            self.blocks = nn.ModuleList(
                [
                    ImpalaCNN.ResidualBlock(out_channels, scale)
                    for _ in range(n_residual_blocks)
                ]
            )

        def forward(self, x: th.Tensor) -> th.Tensor:
            x = self.input_conv(x)
            if self.max_pool:
                x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            for block in self.blocks:
                x = block(x)
            return x
