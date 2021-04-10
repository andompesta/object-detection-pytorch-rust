import torch
from torch import nn
from typing import List

from python.src.utils import ShapeSpec
from python.src.models import InitModule, BuildModule


class FastRCNNConvFCHead(InitModule, BuildModule):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    def __init__(
            self,
            conv_shapes: List[ShapeSpec],
            fc_dims: List[int],
            norm: str = "",
            use_bias: bool = False
    ):
        super().__init__()
        self.norm = norm
        self.use_bias = use_bias
        self.convs = []
        self.norms = []
        self.fcs = []

        for idx, conv_shape in enumerate(conv_shapes):
            conv = nn.Sequential(
                nn.Conv2d(
                    conv_shape.in_channels,
                    conv_shape.out_channels,
                    kernel_size=conv_shape.kernel_size,
                    stride=conv_shape.stride,
                    padding=conv_shape.padding,
                    bias=self.use_bias,
                ),
            )

            self.add_module(f"conv{idx + 1}", conv)
            if self.norm == "BN":
                self.add_module(f"bn{idx + 1}", nn.BatchNorm2d(conv_shape.out_channels))
            else:
                raise NotImplementedError()
            self.add_module(f"relu{idx + 1}", nn.ReLU())


            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])