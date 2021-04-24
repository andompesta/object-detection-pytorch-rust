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
            raise NotImplementedError(f"bottom up architecture {type(conf.bottom_up_conf)} not implemented yet")

        input_shapes = self.bottom_up.output_shapes()

        self.in_features = conf.in_features
        self._out_features = []
        self._in_out_features = OrderedDict()
        self._layers_map = OrderedDict()

        in_channels_per_feature = [input_shapes[f].out_channels for f in self.in_features]

        for idx, (layer_cfg, in_channels, bottom_up_name) in enumerate(zip(
                conf.layers,
                in_channels_per_feature,
                self.bottom_up.layers_name
        )):
            assert layer_cfg.lateral_shape.in_channels == in_channels, \
                f"Miss-match bottom_up shape {in_channels} \t vs \t topdown shape {layer_cfg.lateral_shape.in_channels}"

            name = f"fpn{idx + 1}"
            self.add_module(name, FPNTopDownBlock.build(layer_cfg))
            self._layers_map[bottom_up_name] = name
            self._in_out_features[bottom_up_name] = f"p{idx+1}"
            if f"p{idx+1}" in conf.out_features:
                self._out_features.append(f"p{idx+1}")

        self.top_block = LastLevelMaxPool.build(conf.top_block)
        assert self.top_block.in_feature == self._in_out_features[next(reversed(self._in_out_features))]
        self._in_out_features[self.top_block.in_feature] = f"p{idx + 2}"
        if f"p{idx + 2}" in conf.out_features:
            self._out_features.append(f"p{idx + 2}")


        strides = dict([(self._in_out_features[f], input_shapes[f].stride) for f in self.in_features])
        for s in range(idx + 1, idx + 1 + self.top_block.num_levels):
            strides[self._in_out_features[self.top_block.in_feature]] = 2 ** (s + 1)

        self._out_feature_strides = strides
        self._out_feature_channels = {k: conf.out_channels for k in self._out_features}
        self._fuse_type = conf.fuse_type


    def forward(
            self,
            x: Tensor
    ) -> Dict[str, Tensor]:
        bottom_up_outputs = self.bottom_up(x)
        bottom_up_outputs = OrderedDict(reversed(bottom_up_outputs.items()))
        results = OrderedDict()

        prev = None
        for idx, (bottom_up_name, bottom_up_output) in enumerate(bottom_up_outputs.items()):
            top_down_name = self._layers_map[bottom_up_name]
            top_down_layer = getattr(self, top_down_name)
            x, prev = top_down_layer(bottom_up_output, prev)
            results[self._in_out_features[bottom_up_name]] = x

        top_block_in_feature = bottom_up_outputs.get(self.top_block.in_feature, None)
        if top_block_in_feature is None:
            top_block_in_feature = results[self.top_block.in_feature]
        results[self._in_out_features[self.top_block.in_feature]] = self.top_block(top_block_in_feature)

        for o_feature in self._out_features:
            assert o_feature in results

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