from torch import nn
from abc import ABCMeta
from typing import Dict
from python.src.utils import ShapeSpec

class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()
        self._out_features = []
        self._out_feature_strides = {}
        self._out_feature_channels = {}

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shapes(self) -> Dict[str, ShapeSpec]:
        ...