from typing import Tuple, Union

import torch
from torch import nn, Tensor

__all__ = ['ExponentialMovingAverages']


class ExponentialMovingAverages(nn.Module):
    """TODO docstring
    Implementation adopted from [1].

    Uses bias correction similar to the Adam optimizer, if bias_correction is set to True, otherwise it uses the EMA
    implementation from the original VQ-VAE paper.

    References:
        [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/moving_averages.py
    """

    def __init__(
            self,
            shape_or_initial_value: Union[Tuple[int, ...], Tensor],
            decay: float,
            bias_correction: bool) -> None:
        super().__init__()
        self.register_buffer('_decay', torch.tensor(decay, dtype=torch.float64))
        self._bias_correction = bias_correction

        if isinstance(shape_or_initial_value, Tensor):
            shape = shape_or_initial_value.shape
            initial_value = shape_or_initial_value
        else:
            shape = shape_or_initial_value
            initial_value = None

        if bias_correction:
            if initial_value is None:
                initial_value = torch.zeros(shape)

            self.register_buffer('_average', torch.zeros(shape))
            self.register_buffer('_values', initial_value)
            self.register_buffer('_counter', torch.zeros(1, dtype=torch.long))
        else:
            if initial_value is None:
                initial_value = torch.randn(shape)  # TODO

            self.register_buffer('_average', initial_value)

    @property
    def decay(self) -> float:
        """Returns the value of the decay parameter."""
        return self._decay.item()

    @decay.setter
    def decay(self, value: float) -> None:
        """Sets the value of the decay parameter."""
        self._decay.copy_(torch.tensor(value))

    @property
    def average(self) -> Tensor:
        """Returns the current exponential moving average."""
        return self._average

    def update(self, new_values: Tensor) -> Tensor:
        """Updates the exponential moving average, and returns the new average.

        Arguments:
            new_values (Tensor): Tensor containing the new values.

        Returns:
            Tensor containing the new exponential moving average.
        """
        if self._bias_correction:
            self._counter += 1
            self._values -= (self._values - new_values) * (1 - self._decay)
            self._average.copy_(self._values / (1 - torch.pow(self._decay, self._counter[0]).float()))
        else:
            self._average.copy_(self._decay * self._average + (1 - self._decay) * new_values)
        return self._average

    def copy_(self, new_values: Tensor) -> None:
        self._average.copy_(new_values)
