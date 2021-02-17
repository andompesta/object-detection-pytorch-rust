import torch
import math

from torch import nn
from typing import List, Union, Tuple
from collections.abc import Sequence

from python.src.config import AnchorGeneratorConf
from python.src.models import Boxes

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(
        size: List[int],
        stride: int,
        offset: float,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param size: feature map size
    :param stride: anchor boxes stride
    :param offset: anchor boxes offset
    :param device: device of the feature maps
    :return: anchor box shifts and tilte over the image
    """
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(
        params: Union[List[List[float]], List[float]],
        num_features: int,
        name: str
) -> List[List[float]]:
    """
    Broadcast an the anchors of the size specified in the parameters over multiple features maps. If params is
        list[float], or list[list[float]] with len(params) == 1, repeat it num_features time.
    :param params: sizes (or aspect ratios) to broadcast
    :param num_features: number of time feature maps is repeated
    :param name:
    :return:
    """
    assert isinstance(params, Sequence), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


class DefaultAnchorGenerator(nn.Module):
    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    def __init__(
            self,
            conf: AnchorGeneratorConf
    ):
        """
        Compute anchors in the standard ways described in "Faster R-CNN: Towards Real-Time Object Detection with Region
         Proposal Networks".

        :param conf: configuration of the anchor generator
        """
        super().__init__()

        self.strides = conf.strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(conf.sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(conf.aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = conf.offset
        assert 0.0 <= self.offset < 1.0, self.offset

    def _calculate_anchors(
            self,
            sizes: List[List[float]],
            aspect_ratios: List[List[float]]
    ):
        """
        Generate the initial anchors of given sizes and aspect rations.
        The anchor boxes are defined in XYXY format. The bed are then shifted and tilted over the entire image
        :param sizes: anchor boxes sizes
        :param aspect_ratios: anchor boxes aspect ratios
        :return:
        """
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)
                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(
            self,
            grid_sizes: List[List[int]]
    ):
        """

        :param grid_sizes: feature map sizes
        :return: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(
            self,
            sizes: List[float] = (32, 64, 128, 256, 512),
            aspect_ratios: List[float] = (0.5, 1, 2)
    ) -> torch.Tensor:
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        :param sizes: size of the anchor
        :param aspect_ratios: aspect ratio of the anchor
        :return: Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(
            self,
            features: List[torch.Tensor]
    ) -> List[Boxes]:
        """
        :param features: list of backbone feature maps on which to generate anchors.
        :return: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]