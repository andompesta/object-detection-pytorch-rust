from typing import Union, List, Dict, Tuple
import math
from python.src.config import BaseConf
from python.src.utils import ShapeSpec, RpnLossSpec
from .matcher import MatcherConf

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

_DEFAULT_LOSS_WEIGHTS = RpnLossSpec(1., 1.)

class AnchorGeneratorConf(object):
    def __init__(
            self,
            sizes: Union[List[List[float]], List[float]] = [[32], [64], [128], [256], [512]],
            aspect_ratios: Union[List[List[float]], List[float]] = [[0.5, 1.0, 2.0]],
            offset: float = 0.0,
            box_dim: int = 4
    ):
        """
        :param sizes: If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes (i.e. sqrt of anchor
            area) to use for the i-th feature map. If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
            Anchor sizes are given in absolute lengths in units of the input image; they do not dynamically scale if the
            input image size changes.
        :param aspect_ratios: list of aspect ratios (i.e. height / width) to use for anchors. Same "broadcast" rule for
            `sizes` applies.
        :param offset: Relative offset between the center of the first anchor and the top-left corner of the image.
            Value has to be in [0, 1).
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.offset = offset
        self.box_dim = box_dim


class AnchorMatcherConf(MatcherConf):
    def __init__(
            self,
            iou_thresholds: List[float] = [0.3, 0.7],
            iou_labels: List[int] = [0, -1, 1],
            allow_low_quality_matches: bool = True
    ):
        """
        :param iou_thresholds: a list of Intersection-over-Union thresholds used to stratify predictions into levels.
        :param iou_labels: a list of Intersection-over-Union label predictions belonging at each level. A label can be
            one of {-1, 0, 1} signifying {ignore, negative class, positive class}, respectively.
        :param allow_low_quality_matches: if True, produce additional matches for predictions with maximum match quality
            lower than high_threshold.
            See set_low_quality_matches_ for more details.

        For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.

                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        super(AnchorMatcherConf, self).__init__(
            thresholds=iou_thresholds,
            labels=iou_labels,
            allow_low_quality_matches=allow_low_quality_matches
        )

class Box2BoxTransformConf(object):
    def __init__(
            self,
            weights: Tuple[float, float, float, float] = (1., 1., 1., 1.),
            scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        self.weights = weights
        self.scale_clamp = scale_clamp

class RPNHeadConf(object):
    def __init__(
            self,
            in_channels: int = 64,
            num_anchors: int = 3,
            box_dim: int = 4
    ):
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.conv_shape = ShapeSpec(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.anchor_deltas_shape = ShapeSpec(in_channels=in_channels, out_channels=num_anchors * box_dim, kernel_size=1, stride=1, padding=0, dilation=1)
        self.objectness_logits_shape = ShapeSpec(in_channels=in_channels, out_channels=num_anchors, kernel_size=1, stride=1, padding=0, dilation=1)


class RegionProposalNetworkConf(BaseConf):
    def __init__(
            self,
            name: str = "region-proposal-network",
            head: RPNHeadConf = RPNHeadConf(),
            anchor_generator: AnchorGeneratorConf = AnchorGeneratorConf(),
            anchor_matcher: AnchorMatcherConf = AnchorMatcherConf(),
            box2box_transform: Box2BoxTransformConf = Box2BoxTransformConf(),
            batch_size_per_image: int = 256,
            positive_fraction: float = 0.5,
            pre_nms_topk: Tuple[float, float] = (12000, 6000),
            post_nms_topk: Tuple[float, float] = (2000, 1000),
            nms_thresh: float = 0.7,
            min_box_size: float = 0.0,
            anchor_boundary_thresh: float = -1.0,
            loss_weight: Union[float, RpnLossSpec] = _DEFAULT_LOSS_WEIGHTS,
            box_reg_loss_type: str = "smooth_l1",
            smooth_l1_beta: float = 0.0,
            in_features: List[str] = ['p1', 'p2', 'p3', 'p4', 'p5'],
            **kwargs
    ):
        super(RegionProposalNetworkConf, self).__init__(name=name)
        self.head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform

        # assert self.head.objectness_logits_shape.out_channels == \
        #        len(self.anchor_generator.sizes) * len(self.anchor_generator.aspect_ratios[0]),\
        #     f"object_logit head output size does not match {self.head.objectness_logits_shape.out_channels} {len(self.anchor_generator.sizes[0]) * len(self.anchor_generator.aspect_ratios[0])}"
        #
        # assert self.head.anchor_deltas_shape.out_channels == \
        #        len(self.anchor_generator.sizes[0]) * len(self.anchor_generator.aspect_ratios[0]) * self.anchor_generator.box_dim, \
        #     f"anchor_deltas_shape head output size does not match {self.head.anchor_deltas_shape.out_channels} \t vs. \t {len(self.anchor_generator.sizes[0]) * len(self.anchor_generator.aspect_ratios[0]) * self.anchor_generator.box_dim}"

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.anchor_boundary_thresh = anchor_boundary_thresh
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.in_features = in_features

        for n, v in kwargs.items():
                setattr(self, n, v)