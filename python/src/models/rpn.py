import torch
from torch import nn
from typing import Tuple, List, Dict, Optional
from einops import rearrange

from .base import InitModule
from .modules import AnchorGenerator, RPNHead
from .components import Matcher, Box2BoxTransform, _dense_box_regression_loss
from python.src.config import RegionProposalNetworkConf, RPNHeadConf, AnchorGeneratorConf
from python.src.structures import Boxes, Instances, ImageList, pairwise_iou
from python.src.utils import nonzero_tuple, cat
from python.src.memory import retry_if_cuda_oom

def subsample_labels(
        labels: torch.Tensor,
        num_samples: int,
        positive_fraction: float,
        bg_label: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return `num_samples` (or fewer, if not enough found) random samples from `labels` which is a mixture of positives &
     negatives. It will try to return as many positives as possible without exceeding `positive_fraction * num_samples`,
     and then try to fill the remaining slots with negatives.

    :param labels: (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
    :param num_samples: The total number of labels with value >= 0 to return. Values that are not sampled will be filled
     with -1 (ignore).
    :param positive_fraction: The number of subsampled labels with values > 0 is
     `min(num_positives, int(positive_fraction * num_samples))`. The number of negatives sampled is
     `min(num_negatives, num_samples - num_positives_sampled)`. In order words, if there are not enough positives, the
     sample is filled with negatives. If there are also not enough negatives, then as many elements are sampled as is
     possible.
    :param bg_label: label index of background ("negative") class.
    :return:
    """

    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


class RegionProposalNetwork(InitModule):
    def __init__(
            self,
            conf: RegionProposalNetworkConf
    ):
        super(RegionProposalNetwork, self).__init__()

        # get head
        if isinstance(conf.head, RPNHeadConf):
            self.head = RPNHead(conf.head)
        else:
            raise NotImplementedError(f"head type {type(conf.head)} not implemented yet")

        # get anchor generator
        if isinstance(conf.anchor_generator, AnchorGeneratorConf):
            self.anchor_generator = AnchorGenerator(conf.anchor_generator)
        else:
            raise NotImplementedError(f"anchor generator type {type(conf.anchor_generator)} not implemented yet")

        self.anchor_matcher = Matcher(conf.anchor_matcher)
        self.box2box_transform = Box2BoxTransform(conf.box2box_transform)

        self.batch_size_per_image = conf.batch_size_per_image
        self.positive_fraction = conf.positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = conf.pre_nms_topk
        self.post_nms_topk = conf.post_nms_topk
        self.nms_thresh = conf.nms_thresh
        self.min_box_size = float(conf.min_box_size)
        self.loss_weight = conf.loss_weight
        self.box_reg_loss_type = conf.box_reg_loss_type
        self.smooth_l1_beta = conf.smooth_l1_beta

        self.anchor_boundary_thresh = conf.anchor_boundary_thresh



    def _subsample_labels(
            self,
            label: torch.Tensor
    ):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        :param label: a vector of -1, 0, 1. Will be modified in-place and returned.
        :return:
        """
        pos_idx, neg_idx = subsample_labels(
            label,
            self.batch_size_per_image,
            self.positive_fraction,
            0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
            self,
            anchors: List[Boxes],
            gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """

        :param anchors: anchors for each feature map
        :param gt_instances: the ground-truth image attributes. It contains information like boxes, masks and labels
        :return: A tuple containing:
            - gt_labels: List of #img tensors. i-th element is a vector of labels whose length is the total number of
            anchors across all feature maps R = sum(Hi * Wi * A). Label values are in {-1, 0, 1}, with meanings:
                * -1 = ignore;
                * 0 = negative class;
                * 1 = positive class.
            - matched_gt_boxes: i-th element is a Rx4 tensor. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as 1.
        """

        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the losses from a set of RPN predictions and their associated ground-truth.

        :param anchors: anchors for each feature map, each has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
        :param pred_objectness_logits: A list of L elements. Element i is a tensor of shape (N, Hi*Wi*A) representing
            the predicted objectness logits for all anchors.
        :param gt_labels: Output of :meth:`label_and_sample_anchors`.
        :param pred_anchor_deltas: A list of L elements. Element i is a tensor of shape (N, Hi*Wi*A, 4 or 5)
            representing the predicted "deltas" used to transform anchors to proposals.
        :param gt_boxes: Output of :meth:`label_and_sample_anchors`.
        :return: A dict mapping from loss name to loss value.
        Loss names are:
            * `cls_loss` for objectness classification
            * `loc_loss` for proposal localization
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = nn.functional.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "cls_loss": objectness_loss / normalizer,
            "loc_loss": localization_loss / normalizer,
        }
        losses = {k: v * getattr(self.loss_weight, k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None,
    ):
        """

        :param images: input images of length `N`
        :param features:
        :param gt_instances:
        :return:
        """
        """
        Args:
            images (ImageList): 
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            # score.permute(0, 2, 3, 1).flatten(1)
            rearrange(score, "n a h w -> n (h w a)")
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            # x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            #     .permute(0, 3, 4, 1, 2)
            #     .flatten(1, -2)

            rearrange(x, "n (a b) h w -> n (h w a) b")
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def predict_proposals(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.
        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


    def _init_weights_(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            raise AttributeError(f"unexpected parameter found \n {m}")
