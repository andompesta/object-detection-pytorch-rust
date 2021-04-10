from python.src.config import BaseConf
from python.src.utils import ShapeSpec

from typing import List
from .res_net import ResNet18Conf


class LastLevelConf(object):
    def __init__(
            self,
            num_levels: int = 1,
            in_feature: str = "p5"
    ):
        self.num_levels = num_levels
        self.in_feature = in_feature

class LastLevelMaxPoolConf(LastLevelConf):
    def __init__(
            self,
            kernel_size: int = 1,
            stride: int = 2,
            padding: int = 0
    ):
        super(LastLevelMaxPoolConf, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class FPNStageConf(object):
    def __init__(
            self,
            lateral_shape: ShapeSpec,
            conv_shape: ShapeSpec,
            use_bias: bool = True,
            norm: str = ""
    ):
        super(FPNStageConf, self).__init__()
        assert use_bias == (norm == "")
        self.use_bias = use_bias
        self.norm = norm

        self.lateral_shape = lateral_shape
        self.conv_shape = conv_shape


class FPN18Conf(BaseConf):
    def __init__(
            self,
            name: str = "fpn-18",
            bottom_up_conf: ResNet18Conf = ResNet18Conf(),
            out_channels: int = 64,
            layers: List[FPNStageConf] = [
                FPNStageConf(
                    lateral_shape=ShapeSpec(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
                    conv_shape=ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                ),
                FPNStageConf(
                    lateral_shape=ShapeSpec(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
                    conv_shape=ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                ),
                FPNStageConf(
                    lateral_shape=ShapeSpec(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
                    conv_shape=ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                ),
                FPNStageConf(
                    lateral_shape=ShapeSpec(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
                    conv_shape=ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                )
            ],
            out_features: List[str] = ['p1', 'p2', 'p3', 'p4', 'p5'],
            top_block: LastLevelConf = LastLevelMaxPoolConf(),
            fuse_type: str = "sum",
            **kwargs
    ):
        super(FPN18Conf, self).__init__(name=name)
        self.bottom_up_conf = bottom_up_conf
        self.out_channels = out_channels
        self.layers = layers

        assert len(self.layers) == len(self.bottom_up_conf.layers)

        self.top_block = top_block
        self.fuse_type = fuse_type

        self.out_features = out_features
        self.in_features = self.bottom_up_conf.out_features

        for n, v in kwargs.items():
                setattr(self, n, v)