import itertools
from typing import Tuple, Optional, Union, Collection

import torch
from torch import nn, Tensor

from layers.masked_conv import MaskedConv2d, raster_scan_mask

__all__ = ['gated_activation', 'GatedPixelCNNLayer', 'GatedPixelCNN']


def gated_activation(x: Tensor, dim: int = 1) -> Tensor:
    input, gate = torch.split(x, x.shape[dim] // 2, dim)
    return torch.tanh(input) * torch.sigmoid(gate)


class GatedPixelCNNLayer(nn.Module):
    """Implementation of Figure 3 from http://www.scottreed.info/files/iclr2017.pdf
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            data_channels: int,
            allow_input: bool,
            skip_connection: bool,
            residual_connection: bool) -> None:
        assert isinstance(kernel_size, int)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.data_channels = data_channels
        self.allow_input = allow_input
        self.skip_connection = skip_connection
        self.residual_connection = residual_connection

        if not allow_input and residual_connection:
            print('Warning: residual_connection should probably be False, if allow_input is False')

        # TODO bias

        self.conv_v_nxn = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(kernel_size // 2 + 1, kernel_size), stride=(1, 1),
            padding=(kernel_size // 2 + 1, kernel_size // 2), bias=True)

        self.conv_v_1x1 = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        if self.allow_input:  # TODO condition
            self.conv_v_h_1x1 = MaskedConv2d(
                raster_scan_mask(2 * out_channels, 2 * out_channels, data_channels, allow_input, kernel_size=(1, 1)),
                2 * out_channels, 2 * out_channels,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        else:
            self.conv_v_h_1x1 = nn.Conv2d(
                2 * out_channels, 2 * out_channels,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.conv_h_1xn = MaskedConv2d(
            raster_scan_mask(in_channels, 2 * out_channels, data_channels, allow_input, (1, kernel_size)),
            in_channels, 2 * out_channels,
            kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size // 2), bias=True)

        self.conv_h_1x1 = MaskedConv2d(
            raster_scan_mask(out_channels, out_channels, data_channels, allow_input, kernel_size=(1, 1)),
            out_channels, out_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        if skip_connection:
            self.conv_h_skip_1x1 = MaskedConv2d(
                raster_scan_mask(out_channels, out_channels, data_channels, allow_input, kernel_size=(1, 1)),
                out_channels, out_channels,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

    def forward(
            self,
            v: Tensor,
            h: Tensor,
            skip: Optional[Tensor] = None,
            cond: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:

        v_nxn = self.conv_v_nxn(v)
        v_out = v_nxn[:, :, 1:-(self.kernel_size // 2 + 1), :]
        v_shifted = v_nxn[:, :, :-(self.kernel_size // 2 + 2), :]

        v_out = v_out + self.conv_v_1x1(v)
        if cond is not None:
            v_out = v_out + cond
        v_out = gated_activation(v_out)

        h_out = self.conv_h_1xn(h)
        h_out = h_out + self.conv_v_h_1x1(v_shifted)
        if cond is not None:
            h_out = h_out + cond
        h_out = gated_activation(h_out)

        if self.skip_connection:
            skip_out = self.conv_h_skip_1x1(h_out)
            skip = skip_out if skip is None else skip + skip_out

        h_out = self.conv_h_1x1(h_out)

        if self.residual_connection:
            # TODO v residual connection?
            h_out = h_out + h

        return v_out, h_out, skip


class GatedPixelCNN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            out_hidden_channels: int,
            kernel_sizes: Tuple[int, ...],
            data_channels: int) -> None:
        super().__init__()
        num_layers = len(kernel_sizes)
        assert num_layers >= 1
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_channels = data_channels

        self.causal_layer = GatedPixelCNNLayer(
            in_channels, hidden_channels, kernel_sizes[0], data_channels,
            allow_input=False, skip_connection=False, residual_connection=False)

        self.gated_layers = nn.ModuleList()
        for i in range(1, num_layers):
            gated_layer = GatedPixelCNNLayer(
                hidden_channels, hidden_channels, kernel_sizes[i], data_channels,
                allow_input=True, skip_connection=True, residual_connection=True)
            self.gated_layers.append(gated_layer)

        # TODO bias
        self.conv_out = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(
                raster_scan_mask(hidden_channels, out_hidden_channels, data_channels, allow_input=True, kernel_size=1),
                hidden_channels, out_hidden_channels,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.ReLU(),
            MaskedConv2d(
                raster_scan_mask(out_hidden_channels, out_channels, data_channels, allow_input=True, kernel_size=1),
                out_hidden_channels, out_channels,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        )

    def forward(self, x: Tensor, conds: Optional[Union[Tensor, Collection[Tensor]]] = None) -> Tensor:
        if conds is None:
            cond_iter = itertools.repeat(None)
        elif isinstance(conds, Tensor):
            cond_iter = itertools.repeat(conds, len(self.gated_layers) + 2)
        else:
            cond_iter = iter(conds)

        v, h, skip = self.causal_layer(v=x, h=x, skip=None, cond=next(cond_iter))

        for layer in self.gated_layers:
            v, h, skip = layer(v, h, skip, next(cond_iter))

        out = skip
        cond = next(cond_iter)
        if cond is not None:
            out = out + cond
        out = self.conv_out(out)
        return out
