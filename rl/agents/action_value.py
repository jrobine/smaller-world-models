from abc import ABC
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rl.agents.base import Agent
from rl.networks.action_value import ActionValueNetwork
from rl.spaces.base import TensorSpace

__all__ = ['ActionValueAgent']


class ActionValueAgent(Agent, ABC):
    """An agent that selects actions based on an action-value network, which outputs action values.

    Arguments:
        network (ActionValueNetwork): The action-value network.
    """

    def __init__(self, network: ActionValueNetwork):
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
            action_value, recurrent_state = self.network.compute_action_value(observation, recurrent_state, train)
            action = torch.argmax(action_value, dim=-1)
            if train:
                return {'action': action, 'action_value': action_value}, recurrent_state
            else:
                return {'action': action}, recurrent_state
