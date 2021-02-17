from torch import nn, Tensor
from torch.nn import functional as F
from python.src.utils import ShapeSpec
from python.src.config import ResNetStageConf
from typing import List
from abc import ABCMeta, abstractmethod


class ResidualBlock(nn.Module, metaclass=ABCMeta):
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
    def build_stage(
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

class ResidualBlock50(ResidualBlock):
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

        for idx, conv_shape in enumerate(self.conv_shapes):
            setattr(self, f"conv{idx + 1}", nn.Conv2d(
                conv_shape.in_channels,
                conv_shape.out_channels,
                kernel_size=conv_shape.kernel_size,
                stride=conv_shape.stride,
                padding=conv_shape.padding,
                bias=self.use_bias,
            ))

            if self.norm == "BN":
                setattr(self, f"bn{idx + 1}", nn.BatchNorm2d(conv_shape.out_channels))
            else:
                raise NotImplementedError()

        if self.conv_shapes[0].in_channels != self.conv_shapes[-1].out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.conv_shapes[0].in_channels,
                    self.conv_shapes[-1].out_channels,
                    kernel_size=1,
                    stride=self.conv_shapes[0].stride,
                    bias=self.use_bias,
                ),
                nn.BatchNorm2d(self.conv_shapes[-1].out_channels)
            )
        else:
            self.downsample = None

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)

        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu(out)
        return out





class ResidualBlock18(ResidualBlock):
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

        for idx, conv_shape in enumerate(self.conv_shapes):
            setattr(self, f"conv{idx+1}", nn.Conv2d(
                conv_shape.in_channels,
                conv_shape.out_channels,
                kernel_size=conv_shape.kernel_size,
                stride=conv_shape.stride,
                padding=conv_shape.padding,
                bias=self.use_bias,
            ))

            if self.norm == "BN":
                setattr(self, f"bn{idx+1}", nn.BatchNorm2d(conv_shape.out_channels))
            else:
                raise NotImplementedError()

        if self.conv_shapes[0].in_channels != self.conv_shapes[-1].out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.conv_shapes[0].in_channels,
                    self.conv_shapes[-1].out_channels,
                    kernel_size=1,
                    stride=self.conv_shapes[0].stride,
                    bias=self.use_bias,
                ),
                nn.BatchNorm2d(self.conv_shapes[-1].out_channels)
            )
        else:
            self.downsample = None

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)

        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu(out)
        return out