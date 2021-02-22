from torch import nn, Tensor
from os import path
import numpy as np
from collections import OrderedDict
from typing import Dict

from python.src.config import ResNet18Conf
from python.src.models import Backbone, BaseModel
from python.src.models.modules import BasicStem, ResidualBlock18
from python.src.utils import ShapeSpec

class ResNet18(BaseModel, Backbone):
    def __init__(
            self,
            conf: ResNet18Conf
    ):
        super(ResNet18, self).__init__(conf)

        self.stem = BasicStem(
            conv_shape=conf.stem_conv_shape,
            pool_shape=conf.stem_pool_shape,
            norm=conf.stem_norm,
            use_bias=conf.stem_use_bias
        )
        
        current_stride = conf.stem_conv_shape.stride * conf.stem_pool_shape.stride
        self._out_feature_strides["stem"] = current_stride
        self._out_feature_channels["stem"] = conf.stem_conv_shape.out_channels
        self._out_features = conf.out_features

        layers = [ResidualBlock18.build(l) for l in conf.layers]
        self.layers_name = []
        for idx, (layer_cfg, layer) in enumerate(zip(conf.layers, layers)):
            name = f"res{idx+1}"
            self.add_module(name, layer)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k[0].stride for k in layer_cfg.block_shapes])
            )
            self._out_feature_channels[name] = curr_channels = layer_cfg.block_shapes[-1][-1].out_channels

            self.layers_name.append(name)

        if hasattr(conf, "num_classes"):
            self.avgpool = nn.AdaptiveAvgPool2d(conf.avgpool_size)
            self.fc = nn.Linear(curr_channels, conf.num_classes)
            self.flatten = nn.Flatten(1)
        
    
    def forward(
            self,
            x: Tensor
    ) -> Dict[str, Tensor]:
        output = OrderedDict()
        x = self.stem(x)

        for name in self.layers_name:
            layer = getattr(self, name)
            x = layer(x)

            if name in self._out_features:
                output[name] = x

        if hasattr(self.conf, "num_classes"):
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
            output["logits"] = x

        return output

    def _init_weights_(self, m: nn.Module):
        if hasattr(m, "init_weights"):
            m.init_weights()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        else:
            print(f"--> forward init for module {m}")

    def output_shapes(self) -> Dict[str, ShapeSpec]:
        return OrderedDict([
            (name, ShapeSpec(
                out_channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
                in_channels=None,
                padding=None,
                dilation=None,
                kernel_size=None
            ))
            for name in self._out_features
        ])
        
if __name__ == '__main__':
    conf = ResNet18Conf(num_classes=1000)

    model = ResNet18(conf)
    model.init_weights()

    print(model)

    import torch
    from dynaconf import settings
    model_hb = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    state_dict = model_hb.state_dict()

    conv1_weight = state_dict.pop("conv1.weight")
    bn1_weight = state_dict.pop("bn1.weight")
    bn1_bias = state_dict.pop("bn1.bias")
    bn1_mean = state_dict.pop("bn1.running_mean")
    bn1_var = state_dict.pop("bn1.running_var")
    state_dict.pop("bn1.num_batches_tracked")

    state_dict["stem.conv.weight"] = conv1_weight
    state_dict["stem.bn.weight"] = bn1_weight
    state_dict["stem.bn.bias"] = bn1_bias
    state_dict["stem.bn.running_mean"] = bn1_mean
    state_dict["stem.bn.running_var"] = bn1_var

    print(model.load_state_dict(state_dict, strict=False))

    model.save(
        path_=path.join(settings.get("ckp_dir"), "import"),
        is_best=False,
        file_name="restnet-18.pth.tar"
    )