from typing import List, Optional, Tuple, Union, Dict
from torch import nn

from python.src.utils import ShapeSpec
from .matcher import ProposalMatcherConf

class ROIAlignConf(object):
    def __init__(
            self,
            output_size: Tuple[int, int],
            spatial_scale: float,
            sampling_ratio: int = 0,
            aligned: bool = True
    ):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ration = sampling_ratio
        self.aligned = aligned

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


class ROIHeadConf(object):
    def __init__(self):


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

    def _init_box_head(
            self,
            input_shape: Dict[str, ShapeSpec]
    ):
        # fmt: off
        in_features = self.in_features
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }
