import json
import pickle
from typing import List, Dict
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat

import shutil
from os import path, makedirs
from collections import namedtuple
import cv2

ShapeSpec = namedtuple("ShapeSpec", [
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
])

RpnLossSpec = namedtuple("RpnLossSpec", [
    "cls_loss",
    "loc_loss"
])

LayerSpec = namedtuple("LayerSpec", [
    "block_shapes",
    "use_bias",
    "norm"
])


def is_tracing():
    if torch.jit.is_scripting():
        return False
    else:
        return torch.__version__[:3] >= '1.7' and torch.jit.is_tracing()

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)

def ensure_dir(path_: str) -> str:
    dir = path.dirname(path_)
    if not path.exists(dir):
        makedirs(dir)
    return path_

def save_obj_to_file(path_:str, obj:object):
    with open(ensure_dir(path_), "wb") as writer:
        pickle.dump(obj, writer, protocol=2)

def load_obj_from_file(path_: str) -> object:
    with open(path_, "rb") as reader:
        obj = pickle.load(reader)
    return obj

def save_data_to_json(path_:str, data: object):
    with open(ensure_dir(path_), "w", encoding="utf-8") as w:
        json.dump(data, w, indent=2, sort_keys=True, default=lambda o: o.__dict__)

def load_data_from_json(path_:str) -> object:
    with open(path_, "r", encoding="utf-8") as r:
        return json.load(r)

def save_checkpoint(path_:str, state: Dict, is_best: bool, filename="checkpoint.pth.tar"):
    torch.save(state, ensure_dir(path.join(path_, filename)))
    if is_best:
        shutil.copy(path.join(path_, filename), path.join(path_, "model_best.pth.tar"))

def show_image(path_: str):
    im = cv2.imread(path_)
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im