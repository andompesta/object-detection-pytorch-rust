from typing import Union, List, Dict, Tuple
from python.src.config import BaseConf
from python.src.utils import ShapeSpec


class AnchorGeneratorConf(object):
    def __init__(
            self,
            strides: List[int] = (16),
            sizes: Union[List[List[float]], List[float]] = [[32, 64, 128, 256, 512]],
            aspect_ratios: Union[List[List[float]], List[float]] = [[0.5, 1.0, 2.0]],
            offset: float = 0.0
    ):
        """
        :param sizes: If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes (i.e. sqrt of anchor
            area) to use for the i-th feature map. If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
            Anchor sizes are given in absolute lengths in units of the input image; they do not dynamically scale if the
            input image size changes.
        :param aspect_ratios: list of aspect ratios (i.e. height / width) to use for anchors. Same "broadcast" rule for
            `sizes` applies.
        :param strides: stride of each input feature.
        :param offset: Relative offset between the center of the first anchor and the top-left corner of the image.
            Value has to be in [0, 1).
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.offset = offset

class AnchorMatcherConf(object):
    def __init__(
            self,
            thresholds: List[float] = [0.3, 0.7],
            labels: List[int] = [0, -1, 1],
            allow_low_quality_matches: bool = True
    ):
        """
        :param thresholds: a list of thresholds used to stratify predictions into levels.
        :param labels: a list of values to label predictions belonging at each level. A label can be one of {-1, 0, 1}
            signifying {ignore, negative class, positive class}, respectively.
        :param allow_low_quality_matches: if True, produce additional matches for predictions with maximum match quality
            lower than high_threshold.
            See set_low_quality_matches_ for more details.

        For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches


class RegionProposalNetworkConf(BaseConf):
    def __init__(
            self,
            name: str = "region-proposal-network",
            anchor_generator: AnchorGeneratorConf = AnchorGeneratorConf(),
            anchor_matcher: AnchorMatcherConf = AnchorMatcherConf(),
            batch_size_per_image: int = 256,
            positive_fraction: float = 0.5,
            pre_nms_topk: Tuple[float, float] = (12000, 6000),
            post_nms_topk: Tuple[float, float] = (2000, 1000),
            nms_thresh: float = 0.7,
            min_box_size: float = 0.0,
            anchor_boundary_thresh: float = -1.0,
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            box_reg_loss_type: str = "smooth_l1",
            smooth_l1_beta: float = 0.0,
            **kwargs
    ):
        super(RegionProposalNetworkConf, self).__init__(name=name)
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
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

        for n, v in kwargs.items():
                setattr(self, n, v)