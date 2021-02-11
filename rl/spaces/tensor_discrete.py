from typing import Any, Optional

import torch
from gym import Space
from gym.spaces import Discrete
from torch import Tensor

from rl.spaces.base import TensorSpace
from rl.spaces.tensor_multi_discrete import TensorMultiDiscrete

__all__ = ['TensorDiscrete']


class TensorDiscrete(TensorSpace):
    """Tensor space variant of :class:`gym.spaces.Discrete`.

    Arguments:
        n (int): The number of categories of this space.
        device (Tensor, optional): The device of the elements of this space. Defaults to torch.device('cpu').
    """

    def __init__(self, n: int, device: Optional[torch.device] = None) -> None:
        assert n >= 0
        if device is None:
            device = torch.device('cpu')
        super().__init__(torch.Size(), torch.long, device)
        self.n = n

    @staticmethod
    def from_gym(space: Discrete, device: Optional[torch.device] = None) -> 'TensorDiscrete':
        """Utility method that converts a gym Discrete space to a discrete tensor space.

        Arguments:
            space (Discrete): The gym space used for reference.
            device (torch.device, optional): The device of the tensor space. Defaults to ``None``.
        """
        return TensorDiscrete(space.n, device)

    def sample(self) -> Tensor:
        return (torch.rand(1, generator=self.torch_random, device=self.device)[0] * self.n).to(self.dtype)

    def contains(self, x: Any) -> bool:
        if isinstance(x, int):
            int_value = x
        elif isinstance(x, Tensor):
            if x.numel() > 1:
                return False
            int_value = x.item()
        else:
            return False
        return 0 <= int_value < self.n

    def to_jsonable(self, sample_n: Tensor) -> Any:
        raise NotImplementedError()  # TODO

    def from_jsonable(self, sample_n: Any) -> Tensor:
        raise NotImplementedError()  # TODO

    def batched(self, n: int) -> TensorMultiDiscrete:
        nvec = torch.full((n,), self.n, dtype=torch.long, device=self.device)
        return TensorMultiDiscrete(nvec, self.device)

    def __repr__(self) -> str:
        return f'TensorDiscrete({self.n})'

    def __eq__(self, other: Space) -> bool:
        # TODO device?
        return isinstance(other, TensorDiscrete) and self.device == other.device and self.n == other.n
