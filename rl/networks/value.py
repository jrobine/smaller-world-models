from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import nn, Tensor

from rl.spaces.base import TensorSpace

__all__ = ['ValueNetwork']


class ValueNetwork(nn.Module, ABC):
    """Base class for networks that can compute a state-value based on an observation."""

    @property
    @abstractmethod
    def observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this network expects as input."""
        pass

    @property
    @abstractmethod
    def is_recurrent(self) -> bool:
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
    def compute_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the state-values for a given batch of observations.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            Tensor containing the batch of state-values. TODO docstring
        """
        pass
