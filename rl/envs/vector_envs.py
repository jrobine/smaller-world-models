from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym import Space, Env
from gym.vector import SyncVectorEnv, AsyncVectorEnv, VectorEnv
from gym.vector.utils import batch_space, concatenate, create_empty_array

__all__ = ['ImprovedSyncVectorEnv', 'ImprovedAsyncVectorEnv',
           'SingleVectorEnv']


class ImprovedSyncVectorEnv(SyncVectorEnv):
    """The same as :class:`gym.vector.SyncVectorEnv` but the action space is batched correctly, instead of just using a
    :class:`gym.spaces.Tuple`."""

    def __init__(self,
                 env_fns,
                 observation_space: Optional[Space] = None,
                 action_space: Optional[Space] = None,
                 copy: bool = True) -> None:
        super().__init__(env_fns, observation_space, action_space, copy)
        self.action_space = batch_space(self.single_action_space, self.num_envs)


class ImprovedAsyncVectorEnv(AsyncVectorEnv):
    """The same as :class:`gym.vector.AsyncVectorEnv` but the action space is batched correctly, instead of just using a
    :class:`gym.spaces.Tuple`."""

    def __init__(self,
                 env_fns,
                 observation_space: Optional[Space] = None,
                 action_space: Optional[Space] = None,
                 shared_memory: bool = True,
                 copy: bool = True,
                 context: Optional = None,
                 daemon: bool = True,
                 worker: Optional = None) -> None:
        super().__init__(env_fns, observation_space, action_space, shared_memory, copy, context, daemon, worker)
        self.action_space = batch_space(self.single_action_space, self.num_envs)


class SingleVectorEnv(VectorEnv):
    """Environment that turns a non-vector environment to a vector environment.

    Arguments:
        env (gym.Env): The non-vector environment that should be interpreted as a vector environment with a single
            environment.
    """

    def __init__(self, env: Env) -> None:
        assert not isinstance(env, VectorEnv)
        super().__init__(1, env.observation_space, env.action_space)
        self.action_space = batch_space(env.action_space, 1)
        self.env = env
        self._observation = create_empty_array(env.observation_space, n=1, fn=np.zeros)
        self._reward = np.zeros((1,), dtype=np.float64)
        self._terminal = np.zeros((1,), dtype=np.bool_)
        self._action = None

    def reset_wait(self, **kwargs) -> Any:
        observation = self.env.reset()
        self._observation = concatenate([observation], self._observation, self.single_observation_space)
        return self._observation

    def step_async(self, actions) -> None:
        self._action = actions

    def step_wait(self, **kwargs) -> Tuple[
        Any, np.ndarray, np.ndarray, Tuple[Dict[str, Any]]]:
        observation, self._reward[0], self._terminal[0], info = self.env.step(self._action)
        if self._terminal[0]:
            observation = self.env.reset()
        self._observation = concatenate([observation], self._observation, self.single_observation_space)
        return self._observation, np.copy(self._reward), np.copy(self._terminal), (info,)

    def close_extras(self, **kwargs):
        self.env.close()
