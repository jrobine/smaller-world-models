from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from gym import Space
from gym.utils.seeding import create_seed
from torch import Tensor

__all__ = ['TensorSpace']


class TensorSpace(Space, ABC):
    """Base class for a subclass of gym spaces, that return PyTorch tensors instead of numpy arrays. In addition to a
    shape and dtype, a tensor space specifies a device for the tensors.

    Arguments:
        shape (torch.Size, optional): The shape of the elements of this space. Defaults to ``None``.
        dtype (torch.dtype, optional): The dtype of the elements of this space. Defaults to ``None``.
        device (torch.device, optional): The device of the elements of this space. Defaults to ``None``.
    """

    def __init__(self,
                 shape: Optional[torch.Size] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__(None, None)
        self.shape = shape
        self.dtype = dtype
        self._device = device
        self._torch_random = None

    def from_gym_sample(self, x: np.ndarray) -> Tensor:
        """Converts the sample from a numpy array to a tensor. The sample is expected to be from the corresponding gym
        space, e.g. a space of type :class:`rl.spaces.TensorBox` expects samples from a space of type
        :class:`gym.spaces.Box`.

        Arguments:
            x (np.ndarray): The sample from the gym space.

        Returns:
            Tensor containing the converted sample.
        """
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def to_gym_sample(self, x: Tensor) -> np.ndarray:
        """Converts the sample tensor from this space to a numpy array. The sample will be converted to a sample of the
        corresponding numpy space, e.g. a space of type :class:`rl.spaces.TensorBox` will convert samples to a space of
        type :class:`gym.spaces.Box`.

        Arguments:
            x (Tensor): The sample from this tensor space.

        Returns:
            np.ndarray containing the converted sample.
        """
        return x.detach().cpu().numpy()

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the device for samples from this space, or None if no space was specified."""
        return self._device

    @property
    def torch_random(self) -> torch.Generator:
        """Returns the RNG state for this space, that was initialized with a random seed."""
        if self._torch_random is None:
            self.seed()
        return self._torch_random

    @abstractmethod
    def batched(self, n: int) -> 'TensorSpace':
        """Creates a batched version of this tensor space.

        Arguments:
            n (int): The size of the batch.

        Returns:
            A tensor space with a new batch dimension.
        """
        pass

    @abstractmethod
    def sample(self) -> Tensor:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        seed = create_seed(seed, max_bytes=7)
        self._torch_random = torch.Generator(device=self.device)
        self._torch_random.manual_seed(seed)

    @abstractmethod
    def contains(self, x: Any) -> bool:
        pass

    @abstractmethod
    def to_jsonable(self, sample_n: Any) -> Any:
        pass

    @abstractmethod
    def from_jsonable(self, sample_n: Any) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Space) -> bool:
        pass
