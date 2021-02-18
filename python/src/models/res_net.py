from torch import nn, Tensor
from os import path
import numpy as np


from python.src.config import ResNet18Conf
from python.src.models import Backbone, BaseModel
from python.src.models.modules import BasicStem, ResidualBlock18

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

        self._out_feature_strides = dict(
            stem=conf.stem_conv_shape.stride
        )
        current_stride = conf.stem_conv_shape.stride
        self._out_feature_channels = dict(
            stem=conf.stem_conv_shape.out_channels
        )

        layers = [ResidualBlock18.build(l) for l in conf.layers]

        for idx, (layer_cfg, layer) in enumerate(zip(conf.layers, layers)):
            name = f"layer{idx+1}"
            setattr(self, name, layer)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k[0].stride for k in layer_cfg.block_shapes])
            )
            self._out_feature_channels[name] = curr_channels = layer_cfg.block_shapes[-1][-1].out_channels

        if conf.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(conf.avgpool_size)
            self.fc = nn.Linear(curr_channels, conf.num_classes)
            self.flatten = nn.Flatten(1)



    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.conf.num_classes is not None:
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)

        return x

    def _init_weights_(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        else:
            print(f"--> forward init for module {m}")


if __name__ == '__main__':
    conf = ResNet18Conf(num_classes=1000)

    model = ResNet18(conf)
    model.init_weights()

    print(model)

    import torch
    from dynaconf import settings
    from collections import OrderedDict
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