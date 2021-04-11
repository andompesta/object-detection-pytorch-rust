from torch import nn, Tensor
import math
from typing import Dict
from collections import OrderedDict

from .backbone import Backbone
from python.src.models.base import BaseModel
from .res_net import ResNet18
from python.src.config import FPN18Conf, ResNet18Conf
from python.src.models.modules import LastLevelMaxPool, FPNTopDownBlock, Conv2d
from python.src.utils import ShapeSpec

class FPN(BaseModel, Backbone):
    def __init__(
            self,
            conf: FPN18Conf
    ):
        super(FPN, self).__init__(conf)

        if isinstance(conf.bottom_up_conf, ResNet18Conf):
            self.bottom_up = ResNet18.build(
                conf.bottom_up_conf
            )
        else:
            raise NotImplementedError()

        input_shapes = self.bottom_up.output_shapes()

        self.in_features = conf.in_features
        out_features = []
        layers_name = []

        strides = [input_shapes[f].stride for f in self.in_features]
        in_channels_per_feature = [input_shapes[f].out_channels for f in self.in_features]

        # TODO: check if topdown order need to be reversed
        for idx, (layer_cfg, in_channels) in enumerate(zip(conf.layers, in_channels_per_feature)):
            assert layer_cfg.lateral_shape.in_channels == in_channels, \
                f"Miss-match bottom_up shape {in_channels} \t vs \t topdown shape {layer_cfg.lateral_shape.in_channels}"

            name = f"fpn{idx + 1}"
            self.add_module(name, FPNTopDownBlock.build(layer_cfg))
            layers_name.append(name)
            out_features.append(f"p{idx+1}")
            assert out_features[-1] in conf.out_features

        self.top_block = LastLevelMaxPool(conf.top_block)
        out_features.append(f"p{idx + 2}")
        assert out_features[-1] in conf.out_features
        for s in range(idx + 1, idx + 1 + self.top_block.num_levels):
            strides.append(2 ** (s + 1))

        self._out_features = out_features
        self._layer_names = layers_name[::-1]
        self._out_feature_strides = {f"p{i + 1}": s for i, s in enumerate(strides)}
        self._out_feature_channels = {k: conf.out_channels for k in self._out_features}

        self._fuse_type = conf.fuse_type


    def forward(
            self,
            x: Tensor
    ) -> Dict[str, Tensor]:
        bottom_up_features = self.bottom_up(x)
        xs = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = OrderedDict()

        prev = None
        for idx, (name, x) in enumerate(zip(self._layer_names, xs)):
            top_down_layer = getattr(self, name)
            x, prev = top_down_layer(x, prev)
            results[self._out_features[len(self._layer_names) - idx - 1]] = x

        top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
        if top_block_in_feature is None:
            top_block_in_feature = results[next(iter(results))]
        results[self._out_features[-1]] = self.top_block(top_block_in_feature)

        assert len(self._out_features) == len(results.keys())
        return results


    def output_shapes(self):
        return {
            name: ShapeSpec(
                out_channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
                padding=None,
                in_channels=None,
                dilation=None,
                kernel_size=None
            )
            for name in self._out_features
        }

    def _init_weights_(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) or isinstance(module, Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.)
            nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.002)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)

        else:
            print(f"FPN init -> {module}")

    @classmethod
    def build(
            cls,
            conf: FPN18Conf
    ):
        return cls(conf)