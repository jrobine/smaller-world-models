from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import nn, Tensor
from torch.distributions import Distribution

from rl.spaces.base import TensorSpace

__all__ = ['PolicyGradientNetwork']


class PolicyGradientNetwork(nn.Module, ABC):
    """Base class for networks that can compute an action distribution based on an observation."""

    @property
    @abstractmethod
    def observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this network expects as input."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> TensorSpace:
        """Returns a space that describes the actions that this network assumes for the action distribution."""
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
    def compute_action_distribution(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Optional[Tensor]]:
        """Computes the action distribution for a given batch of observations.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently
                used for training. Defaults to ``False``.

        Returns:
            Distribution object that describes the action distribution. TODO docstring
        """
        pass
