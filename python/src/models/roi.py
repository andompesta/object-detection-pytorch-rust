import torch
from torch import nn, Tensor
from typing import Tuple, List, Optional, Dict
import numpy as np

from .utils import add_ground_truth_to_proposals
from python.src.utils import subsample_labels
from python.src.structures import Instances, pairwise_iou, Logs, ImageList
from python.src.models.components import Matcher
from python.src.models.modules import ROIPooler
from python.src.config import ROIConf



class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    def __init__(
            self,
            num_classes: int,
            batch_size_per_image: int,
            positive_fraction: float,
            proposal_matcher: Matcher,
            proposal_append_gt: bool = True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def build(
            cls,
            conf: ROIConf
    ):
        return ROIHeads(
            num_classes=conf.num_classes,
            batch_size_per_image=conf.batch_size_per_image,
            positive_fraction=conf.positive_fraction,
            proposal_matcher=Matcher.build(conf.proposal_matcher),
            proposal_append_gt=conf.proposal_append_gt
        )

    def _sample_proposals(
            self,
            matched_idxs: Tensor,
            matched_labels: Tensor,
            gt_classes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Based on the matching between N proposals and M ground-truth, sample the proposals and set their classification
            labels.

        :param matched_idxs: a vector of length N, each is the best-matched ground-true index in [0, M) for each proposal.
        :param matched_labels: a vector of length N, the matcher's label (one of iou_labels) for each proposal.
        :param gt_classes: a vector of length M.
        :return:
            -  a vector of indices of sampled proposals. Each is in [0, N).
            - a vector of the same length, the classification label for each sampled proposal. Each sample is labeled as
            either a category in [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            labels=gt_classes,
            num_samples=self.batch_size_per_image,
            positive_fraction=self.positive_fraction,
            bg_label=self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self,
            proposals: List[Instances],
            targets: Optional[List[Instances]]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads. It performs box matching between `proposals` and
        `targets`, and assigns training labels to the proposals. It returns ``self.batch_size_per_image`` random samples
        from proposals and ground-truth boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        :param proposals:  length `N` list of `Instances`. The i-th `Instances` contains object proposals for the i-th
            input image, with fields "proposal_boxes" and "objectness_logits".
        :param targets: length `N` list of `Instances`. The i-th `Instances` contains the ground-truth per-instance
            annotations for the i-th input image.  Specify `targets` during training only.
            It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        :return: length `N` list of `Instances`s containing the proposals sampled for training. Each `Instances` has the
            following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to (this is only meaningful if the
                proposal has a label > 0; if label = 0 then the ground-truth box is random)

        """

        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = Logs.get_instance()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """

        :param images:
        :param features: input data as a mapping from feature map name to tensor. Axis 0 represents the number of images
            `N` in the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g.,
            if a feature pyramid is used).
        :param proposals: length `N` list of `Instances`. The i-th `Instances` contains object proposals for the i-th
            input image, with fields "proposal_boxes" and "objectness_logits".
        :param targets: length `N` list of `Instances`. The i-th `Instances` contains the ground-truth per-instance
            annotations for the i-th input image.  Specify `targets` during training only.
            It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        :return:
            - length `N` list of `Instances` containing the detected instances. Returned during inference only; may be
            [] during training.
            - mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()



class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(
            self,
            num_classes: int,
            batch_size_per_image: int,
            positive_fraction: float,
            proposal_matcher: Matcher,
            proposal_append_gt: bool,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            keypoint_in_features: Optional[List[str]] = None,
            keypoint_pooler: Optional[ROIPooler] = None,
            keypoint_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(
            num_classes=num_classes,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            proposal_matcher=proposal_matcher,
            proposal_append_gt=proposal_append_gt
        )

        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes



    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret
