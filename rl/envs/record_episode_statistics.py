from typing import Any, Dict, Tuple, Union

import torch
from gym import Env, Wrapper
from gym.vector import VectorEnv
from torch import Tensor

from rl.spaces.base import TensorSpace

__all__ = ['RecordEpisodeStatistics']


class RecordEpisodeStatistics(Wrapper):
    """Environment wrapper that adds episode information, i.e., the length and the return, to the info dict at the end
    of episodes. Analogous to the gym wrapper with the same name, but expects the observation space to be a
    :class:`rl.spaces.TensorSpace` and also works with vector spaces.

    Arguments:
        env (gym.Env): The environment for which episode statistics will be recorded.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, TensorSpace)
        device = env.observation_space.device
        self._is_vector = isinstance(env.unwrapped, VectorEnv)
        if self._is_vector:
            self._episode_return = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
            self._episode_length = torch.zeros((env.num_envs,), dtype=torch.long, device=device)
        else:
            self._episode_return = torch.zeros(1, dtype=torch.float32, device=device)
            self._episode_length = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self, **kwargs) -> Tensor:
        observation = super().reset(**kwargs)
        torch.nn.init.zeros_(self._episode_return)
        torch.nn.init.zeros_(self._episode_length)
        return observation

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]]:
        observation, reward, terminal, info = super().step(action)
        if self._is_vector:
            self._episode_return += reward
            self._episode_length += 1
            if terminal.any():
                terminal_indices = torch.nonzero(terminal, as_tuple=True)[0]
                for i in terminal_indices:
                    info[i]['episode'] = {'r': self._episode_return[i].item(), 'l': self._episode_length[i].item()}
                info = tuple(info)
                self._episode_return[terminal] = 0.
                self._episode_length[terminal] = 0
        else:
            self._episode_return[0] += reward.item()
            self._episode_length[0] += 1
            if terminal:
                info['episode'] = {'r': self._episode_return.item(), 'l': self._episode_length.item()}
                self._episode_return[0] = 0.
                self._episode_length[0] = 0
        return observation, reward, terminal, info
