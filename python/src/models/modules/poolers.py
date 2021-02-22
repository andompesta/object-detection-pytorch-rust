import torch
import math
from torch import nn
from typing import List, Dict, Tuple, Union

from python.src.utils import nonzero_tuple, cat
from python.src.structures import Boxes
from python.src.config import ROIPoolerConf

from torchvision.ops import RoIPool
from torchvision.ops import roi_align


class ROIAlign(nn.Module):
    def __init__(
            self,
            output_size: Tuple[int, int],
            spatial_scale: float,
            sampling_ratio: int,
            aligned: bool=True
    ):
        """

        :param output_size:  h, w
        :param spatial_scale: scale the input boxes by this number
        :param sampling_ratio: number of inputs samples to take for each output sample. 0 to take samples densely.
        :param aligned: if False, use the legacy implementation in Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.
            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.
            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(
            self,
            input: torch.Tensor,
            rois: torch.Tensor
    ):
        """

        :param input: NCHW images
        :param rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        :return:
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr


def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
) -> torch.Tensor:
    """
    Map each box in `box_lists` to a feature map level index and return the assignment vector.
    :param box_lists: A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
    :param min_level: Smallest feature map level index. The input is considered index 0, the output of stage 1 is
        index 1, and so.
    :param max_level: Largest feature map level index.
    :param canonical_box_size: A canonical box size in pixels (sqrt(box area)).
    :param canonical_level: The feature map level index on which a canonically-sized box should be placed.
    :return: A tensor of length M, where M is the total number of boxes aggregated over all N batch images. The memory
        layout corresponds to the concatenation of boxes from all images. Each element is the feature map index, as an
        offset from `self.min_level`, for the corresponding box (so value i means the box is at `self.min_level + i`).
    """

    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full_like(
        box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
    )
    return cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(
        box_lists: List[Boxes]
) -> torch.Tensor:
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops (see description under Returns).
    :param box_lists: A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
    :return:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    pooler_fmt_boxes = cat(
        [_fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    def __init__(
        self,
        scales: List[float],
        sampling_ratio: int,
        output_size: Union[int, Tuple[int], List[int]],
        type: str,
        canonical_box_size: int = 224,
        canonical_level: int = 4,
    ):
        """
        Region of interest feature map pooler that supports pooling from one or more feature maps.

        :param output_size: output size of the pooled region, e.g., 14 x 14. If tuple or list is given, the length must be 2.
        :param scales: The scale for each low-level pooling op relative to the input image. For a feature map with
            stride s relative to the input image, scale is defined as 1/s. The stride must be power of 2. When there are
            multiple scales, they must form a pyramid, i.e. they must be a monotically decreasing geometric sequence
            with a factor of 1/2.
        :param sampling_ratio: The `sampling_ratio` parameter for the ROIAlign op.
        :param type: Name of the type of pooling operation that should be applied. For instance, "ROIPool" or "ROIAlignV2".
        :param canonical_box_size: A canonical box size in pixels (sqrt(box area)). The default is heuristically
            defined as 224 pixels in the FPN paper (based on ImageNet pre-training).
        :param canonical_level: The feature map level index from which a canonically-sized box should be placed. The
            default is defined as level 4 (stride=16) in the FPN paper, i.e., a box of size 224x224 will be placed on
            the feature with stride=16. The box placement for all boxes will be determined from their sizes w.r.t
            canonical_box_size. For example, a box whose area is 4x that of a canonical box should be used to pool
            features from feature level ``canonical_level+1``. Note that the actual input feature maps given to this
            module may not have sufficiently many levels for the input boxes. If the boxes are too large or too small
            for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    @classmethod
    def build(
            cls,
            conf: ROIPoolerConf,
            scales: List[float],
    ):
        return cls(
            scales=scales,
            output_size=conf.output_size,
            sampling_ratio=conf.sampling_ratio,
            type=conf.type,
            canonical_box_size=conf.canonical_box_size,
            canonical_level=conf.canonical_level
        )


    def forward(
            self,
            x: List[torch.Tensor],
            box_lists: List[Boxes]
    ) -> torch.Tensor:
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.
        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(box_lists, list), "Arguments to pooler must be lists"
        assert (len(x) == num_level_assignments), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(0), f"unequal value, x[0] batch dim 0 is {x[0].size(0)}, but box_list has length {len(box_lists)}"
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size,
                device=x[0].device,
                dtype=x[0].dtype
            )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level
        )

        num_boxes = pooler_fmt_boxes.size(0)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size),
            dtype=dtype,
            device=device
        )

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

        return output