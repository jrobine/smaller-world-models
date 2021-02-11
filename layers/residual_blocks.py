from torch import nn, Tensor

__all__ = ['ResidualBlock31']


class ResidualBlock31(nn.Module):
    """TODO docstring"""

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            batch_norm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=not batch_norm)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, bias=not batch_norm)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(in_channels)

        self._batch_norm = batch_norm

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.bn1(y) if self._batch_norm else y
        y = nn.functional.relu(y)
        y = self.conv2(y)
        y = self.bn2(y) if self._batch_norm else y
        return y + x
