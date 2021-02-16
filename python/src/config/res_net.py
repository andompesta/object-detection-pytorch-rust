""" XLM-RoBERTa configuration """

from python.src.config import BaseConf
from python.src.utils import ShapeSpec, LayerSpec


class ResNet18Conf(BaseConf):
    def __init__(
            self,
            name: str = "resnet-18",
            stem_conv_shape: ShapeSpec = ShapeSpec(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
                                                   dilation=1),
            stem_pool_shape: ShapeSpec = ShapeSpec(kernel_size=3, stride=2, padding=1, in_channels=None,
                                                   out_channels=None, dilation=None),
            stem_norm: str = "BN",
            stem_use_bias: bool = False,
            layers=[
                LayerSpec(
                    block_shapes=(
                            [
                                ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                                ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
                            ], [
                                ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                                ShapeSpec(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
                            ]
                    ),
                    use_bias=False,
                    norm="BN"
                ),
                LayerSpec(
                    block_shapes=(
                            [
                                ShapeSpec(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1),
                                ShapeSpec(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
                            ], [
                                ShapeSpec(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                                ShapeSpec(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
                            ]
                    ),
                    use_bias=False,
                    norm="BN"
                ),
                LayerSpec(
                    block_shapes=(
                            [
                                ShapeSpec(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1),
                                ShapeSpec(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
                            ], [
                                ShapeSpec(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
                                ShapeSpec(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
                            ]
                    ),
                    use_bias=False,
                    norm="BN"
                ),
                LayerSpec(
                    block_shapes=(
                            [
                                ShapeSpec(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1),
                                ShapeSpec(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
                            ], [
                                ShapeSpec(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
                                ShapeSpec(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
                            ]
                    ),
                    use_bias=False,
                    norm="BN"
                )
            ],
            avgpool_size: int = 1,
            **kwargs
    ):
        super(ResNet18Conf, self).__init__(name=name)
        self.stem_conv_shape = stem_conv_shape
        self.stem_pool_shape = stem_pool_shape
        self.stem_norm = stem_norm
        self.stem_use_bias = stem_use_bias

        self.layers = layers

        self.avgpool_size = avgpool_size
        for n, v in kwargs.items():
                setattr(self, n, v)