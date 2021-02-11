from typing import Tuple, Union

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution, Independent, \
    MixtureSameFamily, Normal

__all__ = ['ConvMDN']


class ConvMDN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_size: int,
            num_gaussians: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            bias: bool = True) -> None:
        super().__init__()
        self._out_size = out_size
        self._num_gaussians = num_gaussians
        self.conv = nn.Conv2d(
            in_channels, num_gaussians + num_gaussians * out_size * 2, kernel_size, stride, padding, bias=bias
        )

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    @property
    def num_gaussians(self) -> int:
        return self._num_gaussians

    @property
    def out_size(self) -> int:
        return self._out_size

    def forward(self, x: Tensor) -> Distribution:
        n = self.num_gaussians
        s = self.out_size
        # TODO logvars vs logstds
        mixture_coeff_logits, means, logvars = torch.split(self.conv(x).permute(0, 2, 3, 1), (n, n * s, n * s), dim=-1)
        stds = torch.exp(0.5 * logvars)
        means = means.view(*means.shape[:-1], n, s)
        stds = stds.view(*stds.shape[:-1], n, s)

        mixture = Categorical(logits=mixture_coeff_logits)
        component = Independent(Normal(means, stds), 1)
        # TODO Independent?
        mixture_model = Independent(MixtureSameFamily(mixture, component), 2)
        return mixture_model
