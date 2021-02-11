import math
from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor

__all__ = ['ConvLSTMCell']


class ConvLSTMCell(nn.Module):
    """Implementation of the convolutional LSTM cell as described in [1]. Layer norm and additional forget bias were
    adapted from [2].

    Arguments:
        in_channels (int): The number of input channels.
        hidden_channels (int): The number of channels for the hidden state and cell state.
        height (int): The height of the input/output.
        width (int): The width of the input/output.
        kernel_size (int or tuple): The kernel size used for all convolutions.
        bias (bool): Whether to add a bias after the convolutions.
        layer_norm (bool, optional): Whether to use layer norm after the convolutions. Defaults to ``False``.
        hidden_peephole_type (str): The type of operation to use for the "peephole" connection to the hidden state.
            Must be either 'elementwise', 'conv', or 'none', for elementwise multiplication, convolution, or no peephole
            connection, respectively. Defaults to 'elementwise'.
        cell_peepholes_type (str): The type of operation to use for the "peephole" connections to the cell state. Must
            be either 'elementwise', 'conv', or 'none', for elementwise multiplications, convolutions, or no peephole
            connections, respectively. Defaults to 'elementwise'.
        additional_forget_bias (float, optional): Additional constant bias that gets added to the forget gate.
            Defaults to 1.

    Inputs: input, (h_t, c_t)
        - **input** of shape `(batch, in_channels, height, width)`: Tensor containing the features of the input for each
            element in the batch.
        - **h_t** of shape `(batch, hidden_channels, height, width)`: Tensor containing the hidden states for each
            element in the batch.
        - **c_t** of shape `(batch, hidden_channels, height, width)`: Tensor containing the cell states for each
            element in the batch.

    Outputs: h_{t+1}, c_{t+1}
        - **h_{t+1}** of shape `(batch, hidden_channels, height, width)`: Tensor containing the new hidden states for
            each element in the batch.
        - **c_{t+1}** of shape `(batch, hidden_channels, height, width)`: Tensor containing the new cell states for each
            element in the batch.

    References:
        [1] Shi, Xingjian and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun (2015).
            Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
            Advances in Neural Information Processing Systems 28, 802-810.
            http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
        [2] https://github.com/Yunbo426/predrnn-pp
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 height: int,
                 width: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 bias: bool,
                 layer_norm: bool = False,
                 hidden_peephole_type: str = 'elementwise',
                 cell_peepholes_type: str = 'elementwise',
                 additional_forget_bias: float = 1.) -> None:
        super().__init__()
        hidden_peephole_type = hidden_peephole_type.lower()
        cell_peepholes_type = cell_peepholes_type.lower()
        assert hidden_peephole_type in ('none', 'elementwise', 'conv')
        assert cell_peepholes_type in ('none', 'elementwise', 'conv')

        self._height = height
        self._width = width
        self._bias = bias
        self._layer_norm = layer_norm
        self._hidden_peephole_type = hidden_peephole_type
        self._cell_peepholes_type = cell_peepholes_type
        self._additional_forget_bias = additional_forget_bias

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        stride = (1, 1)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # * 4 for gates i, f, g, o
        self.conv_x = nn.Conv2d(in_channels, hidden_channels * 4, kernel_size, stride, padding, bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size, stride, padding, bias=False)

        if layer_norm:
            self.layer_norm_x = nn.LayerNorm([hidden_channels * 4, height, width])
            self.layer_norm_h = nn.LayerNorm([hidden_channels * 4, height, width])

        def create_bias_tensor(out_channels: int) -> Tensor:
            x = torch.zeros(out_channels, 1, 1)
            stdv = 1. / math.sqrt(out_channels * kernel_size[0] * kernel_size[1])
            torch.nn.init.uniform_(x, -stdv, stdv)
            return x

        if bias:
            self.bias_i = nn.Parameter(create_bias_tensor(hidden_channels))
            self.bias_f = nn.Parameter(create_bias_tensor(hidden_channels))
            self.bias_g = nn.Parameter(create_bias_tensor(hidden_channels))
            self.bias_o = nn.Parameter(create_bias_tensor(hidden_channels))

        if cell_peepholes_type == 'elementwise':
            self.w_c_i = nn.Parameter(torch.zeros(hidden_channels, 1, 1))
            self.w_c_f = nn.Parameter(torch.zeros(hidden_channels, 1, 1))
            torch.nn.init.normal_(self.w_c_i)  # TODO uniform?
            torch.nn.init.normal_(self.w_c_f)
        elif cell_peepholes_type == 'conv':
            # * 2 for gates i, f
            self.conv_c = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size, stride, padding, bias=False)
            if layer_norm:
                self.layer_norm_c = nn.LayerNorm([hidden_channels * 2, height, width])

        if hidden_peephole_type == 'elementwise':
            self.w_c_new_o = nn.Parameter(torch.zeros(hidden_channels, 1, 1))
            torch.nn.init.normal_(self.w_c_new_o)  # TODO uniform?
        elif hidden_peephole_type == 'conv':
            self.conv_c_new_o = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias=False)
            if layer_norm:
                self.layer_norm_c_new_o = nn.LayerNorm([hidden_channels, height, width])

    @property
    def in_channels(self) -> int:
        """Returns the number of input channels."""
        return self.conv_x.in_channels

    @property
    def hidden_channels(self) -> int:
        """Returns the number of channels for the hidden state and cell state.
        """
        return self.conv_h.in_channels

    @property
    def height(self) -> int:
        """Returns the height of the input/output."""
        return self._height

    @property
    def width(self) -> int:
        """Returns the width of the input/output."""
        return self._width

    def init_recurrent_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """Initializes tensors for the hidden state and cell state for a given batch size.

        Arguments:
            batch_size (int): The desired batch size.
            device (torch.device, optional): The device on which the tensors should be created. If None, the device of
                this module will be used.

        Returns:
            Tuple (h, c) of tensors containing the hidden state and cell state.
        """
        if device is None:
            device = next(self.parameters()).device
        shape = (batch_size, self.hidden_channels, self.height, self.width)
        h = torch.zeros(shape, device=device)
        c = torch.zeros(shape, device=device)
        return (h, c)

    def forward(self, input: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x_i, x_f, x_g, x_o = self._compute_xs(input)
        i, f, g, o = x_i, x_f, x_g, x_o
        del input, x_i, x_f, x_g, x_o

        h, c = hidden
        h_i, h_f, h_g, h_o = self._compute_hs(h)
        i, f, g, o = i + h_i, f + h_f, g + h_g, o + h_o
        del hidden, h, h_i, h_f, h_g, h_o

        c_i, c_f = self._compute_cs(c)
        i = i if c_i is None else i + c_i
        f = f if c_f is None else f + c_f
        del c_i, c_f

        if self._bias:
            i = i + self.bias_i.unsqueeze(0)
            f = f + self.bias_f.unsqueeze(0)
            g = g + self.bias_g.unsqueeze(0)
            o = o + self.bias_o.unsqueeze(0)

        f = f + self._additional_forget_bias

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c_new = f * c + i * g
        del c, i, f, g

        c_new_o = self._compute_c_new_o(c_new)
        o = o if c_new_o is None else o + c_new_o
        del c_new_o

        o = torch.sigmoid(o)
        h_new = o * torch.tanh(c_new)

        hidden_new = (h_new, c_new)
        return hidden_new

    def _compute_xs(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x_conv = self.conv_x(x)
        if self._layer_norm:
            x_conv = self.layer_norm_x(x_conv)
        x_i, x_f, x_g, x_o = torch.split(x_conv, self.hidden_channels, dim=1)
        return x_i, x_f, x_g, x_o

    def _compute_hs(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h_conv = self.conv_h(h)
        if self._layer_norm:
            h_conv = self.layer_norm_h(h_conv)
        h_split = torch.split(h_conv, self.hidden_channels, dim=1)
        h_i, h_f, h_g = h_split[:3]
        h_o = h_split[3]
        return h_i, h_f, h_g, h_o

    def _compute_cs(self, c: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if self._cell_peepholes_type == 'elementwise':
            c_i = self.w_c_i.unsqueeze(0) * c
            c_f = self.w_c_f.unsqueeze(0) * c
            return c_i, c_f
        elif self._cell_peepholes_type == 'conv':
            c_conv = self.conv_c(c)
            if self._layer_norm:
                c_conv = self.layer_norm_c(c_conv)
            c_i, c_f = torch.split(c_conv, self.hidden_channels, dim=1)
            return c_i, c_f
        else:
            return None, None

    def _compute_c_new_o(self, c_new: Tensor) -> Optional[Tensor]:
        if self._hidden_peephole_type == 'elementwise':
            c_new_o = self.w_c_new_o.unsqueeze(0) * c_new
            return c_new_o
        elif self._hidden_peephole_type == 'conv':
            c_new_o = self.conv_c_new_o(c_new)
            if self._layer_norm:
                c_new_o = self.layer_norm_c_new_o(c_new_o)
            return c_new_o
        else:
            return None
