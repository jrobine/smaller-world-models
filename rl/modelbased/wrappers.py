from typing import Any, Dict, Tuple

import torch
from gym import Wrapper
from gym.vector import VectorEnv
from torch import Tensor

from rl.agents.base import Agent
from rl.agents.wrappers import AgentObservationWrapper
from rl.modelbased.env_model import EnvModel

__all__ = ['ModelEnvWrapper', 'ModelAgentWrapper']


class ModelEnvWrapper(Wrapper):
    # TODO unused?
    """An environment wrapper that takes real experience and encodes it using a world model, i.e., an object of type
    :class:`rl.modelbased.env_model`. The observation and action space of this wrapper will match the simulation spaces
    of the world model, but batched.

    Arguments:
        real_env (VectorEnv): The real environment that is wrapped.
        env_model (EnvModel): The world model that is used for encoding.
    """

    def __init__(self, real_env: VectorEnv, env_model: EnvModel) -> None:
        assert isinstance(real_env.unwrapped, VectorEnv)
        super().__init__(real_env)
        assert env_model.real_observation_space == real_env.single_observation_space
        assert env_model.real_action_space == real_env.single_action_space
        self.single_observation_space = env_model.simulated_observation_space
        self.single_action_space = env_model.simulated_action_space
        self.observation_space = env_model.simulated_observation_space.batched(real_env.num_envs)
        self.action_space = env_model.simulated_action_space.batched(real_env.num_envs)
        self._env_model = env_model

    @property
    def env_model(self) -> EnvModel:
        """Returns the world model that is used for encoding."""
        return self._env_model

    @property
    def batch_size(self) -> int:
        """Returns the batch size that is used for encoding. It is equal to the number of environments of the underlying
        real vector environment."""
        return self.num_envs

    def reset(self, **kwargs) -> Tensor:
        observation = super().reset(**kwargs)
        with torch.no_grad():
            if self.env_model.training:
                self.env_model.eval()
            observation, _, _, _ = self.env_model.encode(observation, None, None, None)
        return observation

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        observation, reward, terminal, info = super().step(action)
        with torch.no_grad():
            observation, reward, terminal, info = self.env_model.encode(observation, reward, terminal, info)
        return observation, reward, terminal, info


class ModelAgentWrapper(AgentObservationWrapper):
    """An agent observation wrapper that takes real observations and encodes them using a world model, i.e., an object
    of type :class:`rl.modelbased.env_model`. The observations passed to the underlying agent will be in the world
    model's simulation space, but the observation space of this wrapper will match the real observation space.

    Arguments:
        agent (Agent): The agent that is wrapped and expects observations in simulation space.
        env_model (EnvModel): The world model that is used for encoding.
    """

    def __init__(self, agent: Agent, env_model: EnvModel) -> None:
        assert agent.observation_space == env_model.simulated_observation_space
        super().__init__(agent, env_model.real_observation_space)
        self._env_model = env_model

    @property
    def env_model(self) -> EnvModel:
        """Returns the world model that is used for encoding."""
        return self._env_model

    def observation(self, observation: Tensor) -> Tensor:
        with torch.no_grad():
            if self._env_model.training:
                self._env_model.eval()
            observation, _, _, _ = self._env_model.encode(observation, None, None, None)
        return observation
