from torch import nn, Tensor
from torch.nn import functional as F
from python.src.utils import ShapeSpec
from python.src.models import InitModule


class BasicStem(InitModule):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(
            self,
            conv_shape: ShapeSpec,
            pool_shape: ShapeSpec,
            use_bias: bool,
            norm: str
    ):
        super(BasicStem, self).__init__()
        self.conv_shape = conv_shape
        self.pool_shape = pool_shape
        self.conv = nn.Conv2d(
            in_channels=self.conv_shape.in_channels,
            out_channels=self.conv_shape.out_channels,
            kernel_size=self.conv_shape.kernel_size,
            stride=self.conv_shape.stride,
            padding=self.conv_shape.padding,
            bias=use_bias,
            dilation=self.conv_shape.dilation
        )

        if norm == "BN":
            self.bn = nn.BatchNorm2d(self.conv_shape.out_channels)
        else:
            raise NotImplementedError(f"Norm {norm} non implemented in Stem Layer")

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)

        x = F.relu(x)
        x = F.max_pool2d(
            x,
            kernel_size=self.pool_shape.kernel_size,
            stride=self.pool_shape.stride,
            padding=self.pool_shape.padding
        )
        return x

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)

        else:
            raise AttributeError(f"unexpected parameter found \n {module}")