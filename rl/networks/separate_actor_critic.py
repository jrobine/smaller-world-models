from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.distributions import Distribution

from rl.networks.actor_critic import ActorCriticNetwork
from rl.networks.policy_gradient import PolicyGradientNetwork
from rl.networks.value import ValueNetwork

__all__ = ['SeparateActorCriticNetwork']


class SeparateActorCriticNetwork(ActorCriticNetwork, ABC):
    """Base class for actor-critic networks that consist of a policy-gradient network (actor) and a separate state-value
    network (critic)."""

    @property
    @abstractmethod
    def actor_network(self) -> PolicyGradientNetwork:
        """Returns the policy-gradient network (actor)."""
        pass

    @property
    @abstractmethod
    def critic_network(self) -> ValueNetwork:
        """Returns the state-value network (critic)."""
        pass

    def compute_action_distribution(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Optional[Tensor]]:
        return self.actor_network.compute_action_distribution(observation, recurrent_state, train)

    def compute_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.critic_network.compute_value(observation, recurrent_state, train)

    def compute_action_distribution_and_value(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Tensor, Optional[Tensor]]:
        action_distribution, recurrent_state = self.compute_action_distribution(observation, recurrent_state, train)
        value, recurrent_state = self.compute_value(observation, recurrent_state, train)
        return action_distribution, value, recurrent_state
