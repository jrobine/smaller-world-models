from typing import Any, Dict, Tuple

from gym import Env, Wrapper

__all__ = ['RemoveALEInfo']


class RemoveALEInfo(Wrapper):
    """Environment wrapper that removes the 'ale.lives' item from the info dict, which is produced by Atari
    environments. Does not work with vector environments.

    Arguments:
        env (gym.Env): The environment from which the 'ale.lives' info items will be removed.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def step(self, action: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        observation, reward, done, info = super().step(action)
        info.pop('ale.lives')
        return observation, reward, done, info
