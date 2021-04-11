from torch import nn, Tensor
from torch.nn import functional as F
from python.src.utils import ShapeSpec
from python.src.config import ResNetStageConf
from python.src.models import InitModule, BuildModule
from typing import List
from .wrappers import Conv2d, get_norm


class ResidualBlock(BuildModule):
    def __init__(
            self,
            conv_shapes: List[ShapeSpec],
            norm: str,
            use_bias: bool
    ):
        super().__init__()
        self.conv_shapes = conv_shapes
        self.use_bias = use_bias
        self.norm = norm

    @classmethod
    def build(
            cls,
            stage_cfg: ResNetStageConf
    ):
        blocks = [
            cls(
                conv_shapes=conv_shapes,
                norm=stage_cfg.norm,
                use_bias=stage_cfg.use_bias
            ) for conv_shapes in stage_cfg.block_shapes
        ]
        return nn.Sequential(*blocks)

class ResidualBlock50(ResidualBlock, InitModule):
    """
        The standard bottleneck residual block used by ResNet-18.
        It contains 2 conv layers with kernels
        3x3, 3x3, and a projection shortcut if needed.
        """

    def __init__(
            self,
            conv_shapes: List[ShapeSpec],
            norm: str,
            use_bias: bool
    ):
        super().__init__(conv_shapes=conv_shapes, use_bias=use_bias, norm=norm)

        if norm == "BN":
            assert not use_bias

        for idx, conv_shape in enumerate(self.conv_shapes):
            norm = get_norm(norm, conv_shape.out_channels)
            setattr(self, f"conv{idx + 1}", Conv2d(
                conv_shape.in_channels,
                conv_shape.out_channels,
                kernel_size=conv_shape.kernel_size,
                stride=conv_shape.stride,
                padding=conv_shape.padding,
                bias=self.use_bias,
                norm=norm
            ))

        if self.conv_shapes[0].in_channels != self.conv_shapes[-1].out_channels:
            norm = get_norm(norm, self.conv_shapes[-1].out_channels)
            self.downsample = Conv2d(
                in_channels=self.conv_shapes[0].in_channels,
                out_channels=self.conv_shapes[-1].out_channels,
                kernel_size=1,
                stride=self.conv_shapes[0].stride,
                bias=self.use_bias,
                norm=norm
            )
        else:
            self.downsample = None

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) or isinstance(module, Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)


    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class ResidualBlock18(ResidualBlock, InitModule):
    """
    The standard bottleneck residual block used by ResNet-18.
    It contains 2 conv layers with kernels
    3x3, 3x3, and a projection shortcut if needed.
    """

    def __init__(
            self,
            conv_shapes: List[ShapeSpec],
            norm: str,
            use_bias: bool
    ):
        super().__init__(
            conv_shapes=conv_shapes,
            use_bias=use_bias,
            norm=norm
        )

        for idx, conv_shape in enumerate(self.conv_shapes):
            setattr(self, f"conv{idx+1}", Conv2d(
                in_channels=conv_shape.in_channels,
                out_channels=conv_shape.out_channels,
                kernel_size=conv_shape.kernel_size,
                stride=conv_shape.stride,
                padding=conv_shape.padding,
                bias=self.use_bias,
                norm=get_norm(self.norm, conv_shape.out_channels),
            ))

        if self.conv_shapes[0].in_channels != self.conv_shapes[-1].out_channels:
            self.downsample = Conv2d(
                self.conv_shapes[0].in_channels,
                self.conv_shapes[-1].out_channels,
                kernel_size=1,
                stride=self.conv_shapes[0].stride,
                bias=self.use_bias,
                norm=get_norm(norm, self.conv_shapes[-1].out_channels)
            )
        else:
            self.downsample = None

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) or isinstance(module, Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)

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
        self.conv = Conv2d(
            in_channels=self.conv_shape.in_channels,
            out_channels=self.conv_shape.out_channels,
            kernel_size=self.conv_shape.kernel_size,
            stride=self.conv_shape.stride,
            padding=self.conv_shape.padding,
            bias=use_bias,
            dilation=self.conv_shape.dilation,
            norm=get_norm(norm, self.conv_shape.out_channels),
            activation=nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.max_pool2d(
            x,
            kernel_size=self.pool_shape.kernel_size,
            stride=self.pool_shape.stride,
            padding=self.pool_shape.padding
        )
        return x

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) or isinstance(module, Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)