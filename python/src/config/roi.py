from typing import List, Optional, Tuple, Union
from torch import nn

from .matcher import MatcherConf
from python.src.utils import ShapeSpec

class ROIPoolerConf(object):
    def __init__(
            self,
            output_size: Union[int, Tuple[int, int]],
            type: str = "ROIAlignV2",
            canonical_box_size: int = 224,
            canonical_level: int = 4,
            sampling_ratio: int = 0
    ):
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.output_size = output_size
        self.type = type
        self.sampling_ratio = sampling_ratio


class ProposalMatcherConf(MatcherConf):
    def __init__(
            self,
            roi_thresholds=[0.5],
            roi_labels=[0, 1],
            allow_low_quality_matches=False
    ):
        super(ProposalMatcherConf, self).__init__(
            thresholds=roi_thresholds,
            labels=roi_labels,
            allow_low_quality_matches=allow_low_quality_matches
        )


# class ROIBoxHead(object):
#     def __init__(
#             self,
#             layers = ShapeSpec()
#     ):

class ROIConf(object):
    def __init__(
            self,
            batch_size_per_image: int = 512,
            positive_fraction: float = 0.25,
            num_classes: int = 80,
            proposal_append_gt: bool = True,
            proposal_matcher: ProposalMatcherConf = ProposalMatcherConf(),
            train_on_pred_boxes: bool = False,
            in_features: List[str] = ['p1', 'p2', 'p3', 'p4'],
            box_pooler: ROIPoolerConf = ROIPoolerConf(output_size=(7, 7))
    ):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_append_gt = proposal_append_gt
        self.proposal_matcher = proposal_matcher
        self.train_on_pred_boxes = train_on_pred_boxes
        self.in_features = in_features
        self.box_pooler = box_pooler
