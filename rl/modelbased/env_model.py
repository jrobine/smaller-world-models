from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

from torch import nn, Tensor

from rl.spaces.base import TensorSpace

__all__ = ['EnvModel']

H = TypeVar('H')


class EnvModel(nn.Module, Generic[H], ABC):
    """Base class for world models of a RL environment. World models have to provide the following functionality:
        - Simulation: The world model can be used to simulate experience, which must not be in the same spaces as the
            real experience, e.g., the observation space can be different. This is implemented in `simulate_reset()` and
            `simulate_step()`.
        - Encoding: The world model can be used to encode real experience to match the spaces of the simulated
            experience.

    Arguments:
        H: The type of recurrent states that this world model uses.
    """

    @property
    @abstractmethod
    def real_observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this world model expects as input from the real
        environment (not including the batch size)."""
        pass

    @property
    @abstractmethod
    def real_action_space(self) -> TensorSpace:
        """Returns a space that describes the actions that this world model expects as input from the real environment
        (not including the batch size)."""
        pass

    @property
    @abstractmethod
    def simulated_observation_space(self) -> TensorSpace:
        """Returns a space that describes the observations that this world model produces as output (not including the
        batch size)."""
        pass

    @property
    @abstractmethod
    def simulated_action_space(self) -> TensorSpace:
        """Returns a space that describes the actions that this world model produces as output (not including the batch
        size)."""
        pass

    @abstractmethod
    def simulate_reset(self, batch_size: int) -> Tuple[Tensor, H]:
        """Creates an initial observation in simulation space and an initial recurrent state, with the given batch size.

        Arguments:
            batch_size (int): The batch size of the observation and the recurrent state.

        Returns:
            Tuple of the observation tensor and the recurrent state.
        """
        pass

    @abstractmethod
    def simulate_step(
            self,
            last_observation: Tensor,
            last_reward: Tensor,
            last_terminal: Tensor,
            action: Tensor,
            recurrent_state: H
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Dict[str, Any], ...], H]:
        """Computes the next observation, reward, terminal, info, and recurrent state in simulation space, given the
        values from the last time step and based on the provided action.

        Arguments:
            last_observation (Tensor): Tensor containing the last simulated observation.
            last_reward (Tensor): Tensor containing the last simulated reward.
            last_terminal (Tensor): Tensor containing the last simulated terminal.
            action (Tensor): Tensor containing the action to take.
            recurrent_state (H): The last recurrent state.

        Returns:
            Tuple of (observation, reward, terminal, info, recurrent_state) containing the computed values at the next
            time step.
        """
        pass

    @abstractmethod
    def encode(
            self,
            observation: Optional[Tensor],
            reward: Optional[Tensor],
            terminal: Optional[Tensor],
            info: Optional[Tuple[Dict[str, Any], ...]],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tuple[Dict[str, Any], ...]]]:
        """Encodes the provided observation, reward, terminal, and info from the real space to simulation space. All
        arguments are optional, e.g., to only encode an observation.

        Arguments:
            observation (Tensor, optional): The observation to be encoded.
            reward (Tensor, optional): The reward to be encoded.
            terminal (Tensor, optional): The terminal to be encoded.
            info (tuple of dicts, optional): The info to be encoded.

        Returns:
            Tuple of (observation, reward, terminal, info) containing the encoded values, or ``None``, if the
            corresponding argument was ``None``.
        """
        pass
