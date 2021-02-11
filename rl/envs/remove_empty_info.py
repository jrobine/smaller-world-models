from typing import Any, Dict, Tuple, Union

from gym import Env, Wrapper
from gym.vector import VectorEnv

__all__ = ['RemoveEmptyInfo']


class RemoveEmptyInfo(Wrapper):
    """Environment wrapper that replaces all empty info dicts with the same empty dict instance to save memory. Since
    the same placeholder instance is used, you have to make sure that the returned info dict is not modified. Also
    works with vector environments.

    Arguments:
        env (gym.Env): The environment for which empty info dicts will be replaced with the same empty placeholder dict.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._is_vector = isinstance(env.unwrapped, VectorEnv)
        self._empty = {}

    def step(self, action: Any) -> Tuple[Any, Any, Any, Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]]:
        observation, reward, terminal, info = super().step(action)
        if self._is_vector:
            info = tuple((self._empty if len(i) == 0 else i) for i in info)
        else:
            if len(info) == 0:
                info = self._empty
        return observation, reward, terminal, info
