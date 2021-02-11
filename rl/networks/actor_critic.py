from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.distributions import Distribution

from rl.networks.policy_gradient import PolicyGradientNetwork
from rl.networks.value import ValueNetwork

__all__ = ['ActorCriticNetwork']


class ActorCriticNetwork(PolicyGradientNetwork, ValueNetwork, ABC):
    """Base class for networks that can compute an action distribution (actor) and a state-value (critic) based on an
    observation."""

    @abstractmethod
    def compute_action_distribution_and_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Tensor, Optional[Tensor]]:
        """Computes the action distribution and the state-values for a given batch of observations in a single call.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            Distribution object that describes the action distribution and Tensor containing the batch of state-values.
            TODO docstring
        """
        pass
