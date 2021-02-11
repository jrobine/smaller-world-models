from typing import Any, List, Optional

import torch
from gym import Space
from gym.spaces import Box
from torch import Tensor

from rl.spaces.base import TensorSpace

__all__ = ['TensorBox']


class TensorBox(TensorSpace):
    """Tensor space variant of :class:`gym.spaces.Box`.

    Arguments:
        low (Tensor): A tensor that describes the lower bounds of the elements of this space.
        high (Tensor): A tensor that describes the upper bounds of the elements of this space. Must be of same shape and
            dtype as `low`.
        device (Tensor, optional): The device of the elements of this space. Defaults to the device of `low`.
    """

    def __init__(
            self,
            low: Tensor,
            high: Tensor,
            device: Optional[torch.device] = None) -> None:
        if device is None:
            device = low.device
        super().__init__(low.shape, low.dtype, device)
        assert low.shape == high.shape
        assert low.dtype == high.dtype
        self.low = low.to(self.device)
        self.high = high.to(self.device)

    @staticmethod
    def from_gym(space: Box, device: Optional[torch.device] = None) -> 'TensorBox':
        """Utility method that converts a gym Box space to a box tensor space.

        Arguments:
            space (Box): The gym space used for reference.
            device (torch.device, optional): The device of the tensor space. Defaults to ``None``.
        """
        return TensorBox(
            torch.as_tensor(space.low),
            torch.as_tensor(space.high),
            device)

    def batched(self, n: int) -> 'TensorBox':
        low = self.low.repeat(n, *((1,) * self.low.ndim))
        high = self.high.repeat(n, *((1,) * self.high.ndim))
        return TensorBox(low, high, self.device)

    def sample(self) -> Tensor:
        raise NotImplementedError()  # TODO

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Tensor):
            return False
        return x.shape == self.shape and torch.all(x >= self.low) and torch.all(x <= self.high)

    def to_jsonable(self, sample_n: Tensor) -> list:
        return sample_n.tolist()

    def from_jsonable(self, sample_n: Any) -> List[Tensor]:
        return [torch.as_tensor(sample, dtype=self.dtype, device=self.device) for sample in sample_n]

    def __repr__(self) -> str:
        return f'TensorBox({self.low.min().item()}, {self.high.max().item()},' \
               f' {self.shape}, {self.dtype}, {self.device})'

    def __eq__(self, other: Space) -> bool:
        # TODO device?
        return isinstance(other, TensorBox) and  (self.shape == other.shape) and self.device == other.device and \
               self.dtype == other.dtype and \
               torch.allclose(self.low, other.low) and torch.allclose(self.high, other.high)
