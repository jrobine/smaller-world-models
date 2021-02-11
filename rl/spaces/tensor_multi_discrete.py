from typing import Any, Optional

import torch
from gym import Space
from gym.spaces import MultiDiscrete
from torch import Tensor

from rl.spaces.base import TensorSpace
from rl.spaces.tensor_box import TensorBox

__all__ = ['TensorMultiDiscrete']


class TensorMultiDiscrete(TensorSpace):
    """Tensor space variant of :class:`gym.spaces.MultiDiscrete`.

    Arguments:
        nvec (Tensor): Tensor containing a vector of number of categories.
        device (Tensor, optional): The device of the elements of this space. Defaults to the device of `nvec`.
    """

    def __init__(self, nvec: Tensor, device: Optional[torch.device]) -> None:
        nvec = nvec.long()
        assert torch.all(nvec > 0), 'nvec (counts) have to be positive'
        if device is None:
            device = nvec.device
        super().__init__(nvec.shape, torch.long, device)
        self.nvec = nvec.to(self.device)

    @staticmethod
    def from_gym(space: MultiDiscrete, device: Optional[torch.device] = None) -> 'TensorMultiDiscrete':
        """Utility method that converts a gym MultiDiscrete space to a multi-discrete tensor space.

        Arguments:
            space (MultiDiscrete): The gym space used for reference.
            device (torch.device, optional): The device of the tensor space. Defaults to ``None``.
        """
        return TensorMultiDiscrete(torch.as_tensor(space.nvec), device)

    def sample(self) -> Tensor:
        return (torch.rand(self.nvec.shape, generator=self.torch_random, device=self.device) * self.nvec).to(self.dtype)

    def contains(self, x: Any) -> bool:
        raise NotImplementedError()  # TODO

    def to_jsonable(self, sample_n: Tensor) -> Any:
        raise NotImplementedError()  # TODO

    def from_jsonable(self, sample_n: Any) -> Tensor:
        raise NotImplementedError()  # TODO

    def batched(self, n: int) -> TensorBox:
        raise NotImplementedError()  # TODO

    def __repr__(self):
        return f'TensorMultiDiscrete({str(self.nvec)})'

    def __eq__(self, other: Space) -> bool:
        # TODO device?
        return isinstance(other, TensorMultiDiscrete) and self.device == other.device and \
               torch.all(self.nvec == other.nvec)
