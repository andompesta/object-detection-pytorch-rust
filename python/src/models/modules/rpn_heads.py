from torch import nn, Tensor
from typing import Union, List, Tuple

from python.src.config import RPNHeadConf, RegionProposalNetworkConf
from python.src.models import InitModule, BuildModule
from python.src.utils import ShapeSpec
from .wrappers import Conv2d

class RPNHead(InitModule, BuildModule):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(
            self,
            conv_shape: Union[ShapeSpec, List[ShapeSpec]],
            anchor_deltas_shape: ShapeSpec,
            objectness_logits_shape: ShapeSpec

    ):
        super().__init__()
        self.conv_shape = conv_shape
        self.anchor_deltas_shape = anchor_deltas_shape
        self.objectness_logits_shape = objectness_logits_shape

        if isinstance(self.conv_shape, ShapeSpec):
            self.conv = Conv2d(
                self.conv_shape.in_channels,
                self.conv_shape.out_channels,
                kernel_size=self.conv_shape.kernel_size,
                stride=self.conv_shape.stride,
                padding=self.conv_shape.padding,
                bias=True
            )
        elif isinstance(self.conv_shape, list):
            convs = [
                Conv2d(
                    s.in_channels,
                    s.out_channels,
                    kernel_size=s.kernel_size,
                    stride=s.stride,
                    padding=s.padding,
                    bias=True
                ) for s in self.conv_shape
            ]
            self.conv = nn.Sequential(*convs)
        else:
            raise NotImplementedError()

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = Conv2d(
            self.objectness_logits_shape.in_channels,
            self.objectness_logits_shape.out_channels,
            kernel_size=self.objectness_logits_shape.kernel_size,
            stride=self.objectness_logits_shape.stride,
            padding=self.objectness_logits_shape.padding,
            bias=True
        )

        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = Conv2d(
            self.anchor_deltas_shape.in_channels,
            self.anchor_deltas_shape.out_channels,
            kernel_size=self.anchor_deltas_shape.kernel_size,
            stride=self.anchor_deltas_shape.stride,
            padding=self.anchor_deltas_shape.padding,
            bias=True
        )

    def _init_weights_(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            raise AttributeError(f"unexpected parameter found \n {m}")

    def forward(
            self,
            features: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param features: list of feature maps
        :return:
            - objectness_logits: A list of L elements. Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            - pred_anchor_deltas:  A list of L elements. Element i is a tensor of shape (N, A*box_dim, Hi, Wi)
            representing the predicted "deltas" used to transform anchors to proposals.
        """

        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = nn.functional.relu_(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    @classmethod
    def build(
            cls,
            conf: Union[RegionProposalNetworkConf, RPNHeadConf]
    ):
        if isinstance(conf, RPNHeadConf):
            return cls(
                conv_shape=conf.conv_shape,
                anchor_deltas_shape=conf.anchor_deltas_shape,
                objectness_logits_shape=conf.objectness_logits_shape
            )
        elif isinstance(conf, RegionProposalNetworkConf):
            conf = conf.head
            return cls(
                conv_shape=conf.conv_shape,
                anchor_deltas_shape=conf.anchor_deltas_shape,
                objectness_logits_shape=conf.objectness_logits_shape
            )
        else:
            raise NotImplementedError("configuration not implemented yet")