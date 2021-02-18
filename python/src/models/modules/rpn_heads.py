from torch import nn, Tensor
from python.src.config import RPNHeadConf, RegionProposalNetworkConf
from python.src.models import InitModule
from typing import Union, List, Tuple

class RPNHead(InitModule):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(
            self,
            conf: RPNHeadConf
    ):
        """

        :param in_channels:  number of input feature channels. When using multiple input features, they must have the
            same number of channels.
        :param num_anchors: number of anchors to predict for *each spatial position* on the feature map. The total
            number of anchors for each feature map will be `num_anchors * H * W`.
        :param box_dim: dimension of a box, which is also the number of box regression predictions to make for each
            anchor. An axis aligned box has box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        self.conv_shape = conf.conv_shape
        self.anchor_deltas_shape = conf.anchor_deltas_shape
        self.objectness_logits_shape = conf.objectness_logits_shape

        self.conv = nn.Conv2d(
            self.conv_shape.in_channels,
            self.conv_shape.in_channels,
            kernel_size=self.conv_shape.kernel_size,
            stride=self.conv_shape.stride,
            padding=self.conv_shape.padding
        )

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(
            self.objectness_logits_shape.in_channels,
            self.objectness_logits_shape.out_channels,
            kernel_size=self.objectness_logits_shape.kernel_size,
            stride=self.objectness_logits_shape.stride,
            padding=self.objectness_logits_shape.padding
        )

        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            self.anchor_deltas_shape.in_channels,
            self.anchor_deltas_shape.out_channels,
            kernel_size=self.anchor_deltas_shape.kernel_size,
            stride=self.anchor_deltas_shape.stride,
            padding=self.anchor_deltas_shape.padding
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
            t = nn.functional.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


    @classmethod
    def build(
            cls,
            conf: Union[RegionProposalNetworkConf, RPNHeadConf]
    ):
        if isinstance(conf, RPNHeadConf):
            return cls(conf)
        elif isinstance(conf, RegionProposalNetworkConf):
            return cls(conf.head)
        else:
            raise NotImplementedError("configuration not implemented yet")