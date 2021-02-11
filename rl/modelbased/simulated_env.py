from typing import Any, Dict, Tuple, Optional

import torch
from gym.spaces import Discrete
from gym.vector import VectorEnv
from torch import Tensor

from rl.modelbased.env_model import EnvModel

__all__ = ['SimulatedEnv']


class SimulatedEnv(VectorEnv):
    """An environment that uses a world model, i.e., an object of type :class:`rl.modelbased.EnvModel`, to simulate
    experience. The observation and action space will match the simulation spaces of the world model, but batched.

    Arguments:
        env_model (EnvModel): The world model to use for simulation.
        batch_size (int): The batch size to use for simulation.
        max_episode_steps (int, optional): The maximum number of steps to simulate before resetting.
    """

    def __init__(
            self,
            env_model: EnvModel,
            batch_size: int,
            max_episode_steps: Optional[int] = None) -> None:
        dummy_space = Discrete(1)
        super().__init__(batch_size, dummy_space, dummy_space)
        del dummy_space
        self.observation_space = env_model.simulated_observation_space.batched(batch_size)
        self.action_space = env_model.simulated_action_space.batched(batch_size)
        self.single_observation_space = env_model.simulated_observation_space
        self.single_action_space = env_model.simulated_action_space
        self._env_model = env_model
        self._max_episode_steps = max_episode_steps
        self._last_observation = None
        self._last_reward = None
        self._last_terminal = None
        self._recurrent_state = None
        self._actions = None
        self._elapsed_steps = None

    @property
    def env_model(self) -> EnvModel:
        """Returns the world model that is used for simulation."""
        return self._env_model

    @property
    def batch_size(self) -> int:
        """Returns the batch size that is used for simulation."""
        return self.num_envs

    @property
    def max_episode_steps(self) -> Optional[int]:
        """Returns the maximum number of steps to simulate before resetting, if
        specified, otherwise ``None``."""
        return self._max_episode_steps

    def reset_wait(self) -> Tensor:
        with torch.no_grad():
            if self._env_model.training:
                self._env_model.eval()
            observation, recurrent_state = self._env_model.simulate_reset(self.batch_size)
            self._last_observation = observation
            self._last_reward = torch.zeros(observation.shape[0], dtype=torch.float, device=observation.device)
            self._last_terminal = torch.zeros(observation.shape[0], dtype=torch.bool, device=observation.device)
            self._recurrent_state = recurrent_state
            self._actions = None
            self._elapsed_steps = 0
        return observation

    def step_async(self, actions: Tensor) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[Tensor, Tensor, Tensor, Tuple[Dict[str, Any], ...]]:
        with torch.no_grad():
            if self._env_model.training:
                self._env_model.eval()
            action = self._actions
            observation, reward, terminal, info, recurrent_state = self._env_model.simulate_step(
                    self._last_observation, self._last_reward, self._last_terminal, action, self._recurrent_state)

            self._elapsed_steps += 1
            if self._max_episode_steps is not None and self._elapsed_steps >= self._max_episode_steps:
                observation, recurrent_state = self._env_model.simulate_reset(self.batch_size)
                self._elapsed_steps = 0
                terminal.fill_(True)

        self._last_observation = observation
        self._last_reward = reward
        self._last_terminal = terminal
        self._recurrent_state = recurrent_state
        self._actions = None
        return observation, reward, terminal, info

    def close_extras(self, **kwargs) -> None:
        self._last_observation = None
        self._last_reward = None
        self._last_terminal = None
        self._recurrent_state = None
        self._actions = None
