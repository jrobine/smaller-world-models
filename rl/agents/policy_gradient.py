from abc import ABC
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rl.agents.base import Agent
from rl.networks.policy_gradient import PolicyGradientNetwork
from rl.spaces.base import TensorSpace

__all__ = ['PolicyGradientAgent']


class PolicyGradientAgent(Agent, ABC):
    """An agent that selects actions based on an policy gradient network, which outputs an action distribution.

    Arguments:
        network (PolicyGradientNetwork): The policy gradient network.
    """

    def __init__(self, network: PolicyGradientNetwork):
        super().__init__()
        self.network = network

    @property
    def observation_space(self) -> TensorSpace:
        return self.network.observation_space

    @property
    def action_space(self) -> TensorSpace:
        return self.network.action_space

    def init_recurrent_state(self, batch_size: int) -> Optional[Tensor]:
        return self.network.init_recurrent_state(batch_size)

    def mask_recurrent_state(self, recurrent_state: Optional[Tensor], terminal: Tensor) -> Optional[Tensor]:
        return self.network.mask_recurrent_state(recurrent_state, terminal)

    def act(self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        if self.network.training:
            self.network.eval()
        with torch.no_grad():
            action_distribution, recurrent_state = \
                self.network.compute_action_distribution(observation, recurrent_state, train)
            action = action_distribution.sample()
            if train:
                log_prob = action_distribution.log_prob(action)
                return {'action': action, 'log_prob': log_prob}, recurrent_state
            else:
                return {'action': action_distribution.sample()}, recurrent_state
