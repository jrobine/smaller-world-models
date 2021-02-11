from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import nn, Tensor

from rl.spaces.base import TensorSpace

__all__ = ['ActionValueNetwork']


class ActionValueNetwork(nn.Module, ABC):
    """Base class for networks that can compute action-values based on an observation."""

    @property
    @abstractmethod
    def observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this network expects as input."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> TensorSpace:
        """Returns a space that describes the actions that this network assumes for the action-values."""
        pass

    @abstractmethod
    def compute_action_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the action-values for a given batch of observations.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            Tensor containing the batch of action-values. TODO docstring
        """
        pass
