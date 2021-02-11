from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from torch import Tensor

from rl.agents.base import Agent
from rl.spaces.base import TensorSpace

__all__ = ['AgentObservationWrapper']


class AgentObservationWrapper(Agent, ABC):
    """Base class for wrappers of an :class:`rl.agents.Agent`, that modify the observation that is provided to the
    underlying agent.

    Arguments:
        agent (Agent): The agent that is wrapped.
        observation_space (TensorSpace): The observation space that is expected
            as input to the transform function.
    """

    def __init__(self, agent: Agent, observation_space: TensorSpace) -> None:
        self._agent = agent
        self._observation_space = observation_space

    @property
    def agent(self) -> Agent:
        """Returns the agent that is wrapped."""
        return self._agent

    @property
    def observation_space(self) -> TensorSpace:
        return self._observation_space

    @property
    def action_space(self) -> TensorSpace:
        return self.agent.action_space

    def init_recurrent_state(self, batch_size: int) -> Optional[Tensor]:
        return self.agent.init_recurrent_state(batch_size)

    def mask_recurrent_state(self, recurrent_state: Optional[Tensor], terminal: Tensor) -> Optional[Tensor]:
        return self.agent.mask_recurrent_state(recurrent_state, terminal)

    def act(self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        return self.agent.act(self.observation(observation), recurrent_state, train)

    @abstractmethod
    def observation(self, observation: Tensor) -> Tensor:
        """Transforms an observation from this wrapper's observation space to the underlying agent's observation space.

        Arguments:
            observation (Tensor): The observation to be transformed.

        Returns:
            Tensor containing the transformed observation.
        """
        pass
