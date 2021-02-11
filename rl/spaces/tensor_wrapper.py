from typing import Any, Optional, Tuple

import torch
from gym import Wrapper, Env, Space
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.vector import VectorEnv
from torch import Tensor

from rl.spaces.base import TensorSpace
from rl.spaces.tensor_box import TensorBox
from rl.spaces.tensor_discrete import TensorDiscrete
from rl.spaces.tensor_multi_discrete import TensorMultiDiscrete

__all__ = ['TensorWrapper', 'to_tensor_space']


class TensorWrapper(Wrapper):
    """Wrapper class that converts the spaces of an environment to tensor spaces and accordingly converts the values of
    reset() and step() to tensors.

    Arguments:
        env (Env): The environment that is wrapped. The observation space and action space may not be tensor spaces
            already.
        device (torch.device, optional): The device that is used for the observation and action tensor spaces.
            Defaults to ``None``.
    """

    def __init__(self, env: Env, device: Optional[torch.device] = None) -> None:
        super().__init__(env)
        assert not isinstance(env.observation_space, TensorSpace)
        assert not isinstance(env.action_space, TensorSpace)
        self.observation_space = to_tensor_space(env.observation_space, device)
        self.action_space = to_tensor_space(env.action_space, device)

        if isinstance(env.unwrapped, VectorEnv):
            self.single_observation_space = to_tensor_space(env.unwrapped.single_observation_space, device)
            self.single_action_space = to_tensor_space(env.unwrapped.single_action_space, device)

    def reset(self, **kwargs) -> Tensor:
        obs = self.env.reset(**kwargs)
        obs = self.observation_space.from_gym_sample(obs)
        return obs

    def step(self, action: Any) -> Tuple[Tensor, Tensor, Tensor, dict]:
        action = self.action_space.to_gym_sample(action)
        obs, reward, terminal, info = self.env.step(action)
        obs = self.observation_space.from_gym_sample(obs)
        reward = torch.as_tensor(reward, dtype=torch.float, device=obs.device)
        terminal = torch.as_tensor(terminal, dtype=torch.bool, device=obs.device)
        return obs, reward, terminal, info


def to_tensor_space(space: Space, device: Optional[torch.device] = None) -> TensorSpace:
    """Utility method that converts a gym space to a tensor space. Raises a ValueError if the space cannot be converted
    to a tensor space.

    Arguments:
        space (Space): The gym space to convert.
        device (torch.device, optional): The device for the tensor space. Defaults to ``None``.

    Returns:
        The converted tensor space. If the input already was a tensor space, the space is returned without conversion.
    """
    if isinstance(space, TensorSpace):
        return space
    elif isinstance(space, Box):
        return TensorBox.from_gym(space, device)
    elif isinstance(space, Discrete):
        return TensorDiscrete.from_gym(space, device)
    elif isinstance(space, MultiDiscrete):
        return TensorMultiDiscrete.from_gym(space, device)
    else:
        raise ValueError()
