from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from torch import Tensor

from rl.spaces.base import TensorSpace

__all__ = ['Agent']


class Agent(ABC):
    """Base class for agents, that take an observation and produce an output dictionary, which contains the selected
    action and possibly more information. Agents are expected to only work with tensors, therefore all spaces must be of
    type :class:`rl.spaces.TensorSpace`."""

    @property
    @abstractmethod
    def observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this agent expects as input (not including the batch
        size)."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> TensorSpace:
        """Returns a space that describes the actions that this agent produces as output (not including the batch
        size)."""
        pass

    @abstractmethod
    def init_recurrent_state(self, batch_size: int) -> Optional[Tensor]:
        """TODO docstring"""
        pass

    @abstractmethod
    def mask_recurrent_state(self, recurrent_state: Optional[Tensor], terminal: Tensor) -> Optional[Tensor]:
        """TODO docstring"""
        pass

    @abstractmethod
    def act(self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Dict[str, Tensor], Tensor]:
        """Takes a batch of observations and selects actions.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the agent is currently used for training. For example, this can be
                used to compute log-probabilities only during training. Defaults to ``False``.

        Returns:
            Dictionary with (str, Tensor) items, which has to contain the key 'action' for the selected batch of
            actions. It may contain more items, e.g. the log-probabilities used by policy gradient algorithms.
            TODO docstring
        """
        pass
