from typing import List, Optional, Tuple
from torch import nn, Tensor
from torch.nn import functional as F

from python.src.config import LastLevelMaxPoolConf
from python.src.config import FPNStageConf
from python.src.models import InitModule, BuildModule
from python.src.utils import ShapeSpec
from .wrappers import Conv2d, get_norm

class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(
            self,
            conf: LastLevelMaxPoolConf
    ):
        super().__init__()
        self.num_levels = conf.num_levels
        self.in_feature = conf.in_feature
        self.kernel_size = conf.kernel_size
        self.stride = conf.stride
        self.padding = conf.padding

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

class FPNTopDownBlock(InitModule):
    def __init__(
            self,
            lateral_shape: ShapeSpec,
            output_shape: ShapeSpec,
            use_bias: bool,
            norm: str
    ):
        super(FPNTopDownBlock, self).__init__()
        lateral_norm = get_norm(norm, lateral_shape.out_channels)
        output_norm = get_norm(norm, lateral_shape.out_channels)

        self.lateral = Conv2d(
            in_channels=lateral_shape.in_channels,
            out_channels=lateral_shape.out_channels,
            kernel_size=lateral_shape.kernel_size,
            stride=lateral_shape.stride,
            padding=lateral_shape.padding,
            bias=use_bias,
            norm=lateral_norm,
        )

        self.output = Conv2d(
            output_shape.in_channels,
            output_shape.out_channels,
            kernel_size=output_shape.kernel_size,
            stride=output_shape.stride,
            padding=output_shape.padding,
            bias=use_bias,
            norm=output_norm
        )

        self.use_bias = use_bias
        self.scale_factor = 2

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) or isinstance(module, Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if self.use_bias:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)


    def forward(
            self,
            x: Tensor,
            prev: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        lateral_features = self.lateral(x)

        if prev is not None:
            top_down_features = F.interpolate(prev, scale_factor=self.scale_factor, mode="nearest")
            prev = lateral_features + top_down_features
        else:
            prev = lateral_features

        x = self.output(prev)

        return x, prev

    @classmethod
    def build(
            cls,
            conf: FPNStageConf
    ):
        return cls(
            conf.lateral_shape,
            conf.conv_shape,
            conf.use_bias,
            conf.norm
        )
