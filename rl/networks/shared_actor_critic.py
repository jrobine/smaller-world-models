from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.distributions import Distribution

from rl.networks.actor_critic import ActorCriticNetwork

__all__ = ['SharedActorCriticNetwork']


class SharedActorCriticNetwork(ActorCriticNetwork, ABC):
    """Base class for actor-critic networks that share parameters between the actor and the critic. The shared network
    output is computed in `compute_hidden()`, and the resulting hidden tensor can be used in
    `compute_action_distribution_shared()` and `compute_value_shared()`."""

    @abstractmethod
    def compute_hidden(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the shared hidden tensor that can be used in `compute_action_distribution_shared()` and
        `compute_value_shared()`.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            The hidden tensor. TODO docstring
        """
        pass

    @abstractmethod
    def compute_action_distribution_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Optional[Tensor]]:
        """Computes the action distribution for a given batch of observations.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            hidden (Tensor): The output of `compute_hidden()`.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            Distribution object that describes the action distribution. TODO docstring
        """
        pass

    @abstractmethod
    def compute_value_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the state-values for a given batch of observations.

        Arguments:
            observation (Tensor): A batch of observations of a single time step.
            hidden (Tensor): The output of `compute_hidden()`.
            recurrent_state (Tensor, optional): TODO docstring
            train (bool, optional): Indicates whether the network is currently used for training. Defaults to ``False``.

        Returns:
            Tensor containing the batch of state-values. TODO docstring
        """
        pass

    def compute_action_distribution(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Optional[Tensor]]:
        hidden, recurrent_state = self.compute_hidden(observation, recurrent_state, train)
        return self.compute_action_distribution_shared(observation, hidden, recurrent_state, train)

    def compute_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        hidden, recurrent_state = self.compute_hidden(observation, recurrent_state, train)
        return self.compute_value_shared(observation, hidden, recurrent_state, train)

    def compute_action_distribution_and_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Tensor, Optional[Tensor]]:
        hidden, recurrent_state = self.compute_hidden(observation, recurrent_state, train)
        action_distribution, recurrent_state = \
            self.compute_action_distribution_shared(observation, hidden, recurrent_state, train)
        value, recurrent_state = self.compute_value_shared(observation, hidden, recurrent_state, train)
        return action_distribution, value, recurrent_state
