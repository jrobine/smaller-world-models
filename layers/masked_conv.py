from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor

__all__ = ['MaskedConv2d', 'raster_scan_mask']


class MaskedConv2d(nn.Conv2d):

    def __init__(
            self,
            mask: Tensor,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros') -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


def raster_scan_mask(
        in_channels: int,
        out_channels: int,
        data_channels: int,
        allow_input: bool,
        kernel_size: Union[int, Tuple[int, int]],
        dtype: torch.dtype = torch.bool,
        device: Optional[torch.device] = None) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    h, w = kernel_size

    mask = np.ones((out_channels, in_channels, h, w), np.bool)
    mask[:, :, h // 2, w // 2 + 1:] = False
    mask[:, :, h // 2 + 1:, :] = False

    # https://github.com/anordertoreclaim/PixelCNN/blob/master/pixelcnn/conv_layers.py
    def cmask(in_channel: int, out_channel: int) -> np.ndarray:
        out_mask = torch.arange(out_channels) % data_channels == out_channel
        in_mask = torch.arange(in_channels) % data_channels == in_channel
        return out_mask[:, None] * in_mask[None, :]

    for o in range(data_channels):
        if not allow_input:
            mask[cmask(o, o), h // 2, w // 2] = 0
        for i in range(o + 1, data_channels):
            mask[cmask(i, o), h // 2, w // 2] = 0

    return torch.as_tensor(mask, dtype=dtype, device=device)
