from typing import Any, Dict, Optional, Tuple

import torch
from gym.vector import VectorEnv
from torch import Tensor

from rl.agents.base import Agent
from rl.spaces.base import TensorSpace

__all__ = ['RandomAgent']


class RandomAgent(Agent):
    """An agent that selects actions randomly."""

    def __init__(self, observation_space: TensorSpace, action_space: TensorSpace) -> None:
        super().__init__()
        self._observation_space = observation_space
        self._action_space = action_space

    @staticmethod
    def for_env(env: VectorEnv) -> 'RandomAgent':
        """Utility method to create a random agent based on an environment's observation and action space.

        Arguments:
            env (VectorEnv): The environment that is used for reference.

        Returns:
            The new random agent.
        """
        return RandomAgent(env.single_observation_space, env.single_action_space)

    @property
    def observation_space(self) -> TensorSpace:
        return self._observation_space

    @property
    def action_space(self) -> TensorSpace:
        return self._action_space

    def init_recurrent_state(self, batch_size: int) -> Optional[Tensor]:
        return None

    def mask_recurrent_state(self, recurrent_state: Optional[Tensor], terminal: Tensor) -> Optional[Tensor]:
        return recurrent_state

    def act(self,
            observation: Any,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        batch_size = observation.shape[0]
        x = {'action': torch.stack([self.action_space.sample() for _ in range(batch_size)])}
        recurrent_state = None
        return x, recurrent_state
