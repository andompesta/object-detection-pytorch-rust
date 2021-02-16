from torch import nn, Tensor
from torch.nn import functional as F
from python.src.utils import ShapeSpec, LayerSpec
from typing import List


class ResidualBlock18(nn.Module):
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
        super().__init__()
        self.conv_shapes = conv_shapes
        self.use_bias = use_bias
        self.norm = norm


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
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu(out)
        return out

    @classmethod
    def build_stage(
            cls,
            layer: LayerSpec
    ):
        blocks = [
            cls(
                conv_shapes=conv_shapes,
                norm=layer.norm,
                use_bias=layer.use_bias
            ) for conv_shapes in layer.block_shapes
        ]
        return nn.Sequential(*blocks)