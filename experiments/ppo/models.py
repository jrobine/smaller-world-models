from math import sqrt

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution

from rl.networks import SharedActorCriticNetwork
from rl.spaces import TensorBox, TensorDiscrete, TensorSpace

__all__ = ['NatureCNN', 'AtariActorCriticNetwork']


class NatureCNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc4 = nn.Linear(64 * 7 * 7, 512)

        for layer in (self.conv1, self.conv2, self.conv3, self.fc4):
            nn.init.orthogonal_(layer.weight, sqrt(2.0))
            nn.init.zeros_(layer.bias)

    @property
    def output_size(self) -> int:
        return self.fc4.weight.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc4(x))
        return x


class AtariActorCriticNetwork(SharedActorCriticNetwork):

    def __init__(self, num_actions: int, device: torch.device) -> None:
        super().__init__()
        self._observation_space = TensorBox(
            low=torch.zeros((4, 84, 84), dtype=torch.uint8, device=device),
            high=torch.full((4, 84, 84), 255, dtype=torch.uint8, device=device),
            device=device)
        self._action_space = TensorDiscrete(num_actions, device)
        self.hidden = NatureCNN()
        self.action_logits = nn.Linear(self.hidden.output_size, num_actions)
        self.value = nn.Linear(self.hidden.output_size, 1)
        self.to(device)

    @property
    def observation_space(self) -> TensorSpace:
        return self._observation_space

    @property
    def action_space(self) -> TensorSpace:
        return self._action_space

    def forward(self, observation: Tensor) -> Tensor:
        x = observation.float() / 255.
        return self.hidden(x)

    def compute_hidden(
            self,
            observation: Tensor,
            train: bool = False) -> Tensor:
        return self.forward(observation)

    def compute_action_distribution_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            train: bool = False) -> Distribution:
        logits = self.action_logits(hidden)
        return Categorical(logits=logits)

    def compute_value_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            train: bool = False) -> Tensor:
        return self.value(hidden).squeeze(-1)
