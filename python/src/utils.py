import json
import pickle
import typing
import torch
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


def cat(tensors: typing.List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


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

def save_checkpoint(path_:str, state: typing.Dict, is_best: bool, filename="checkpoint.pth.tar"):
    torch.save(state, ensure_dir(path.join(path_, filename)))
    if is_best:
        shutil.copy(path.join(path_, filename), path.join(path_, "model_best.pth.tar"))

def show_image(path_: str):
    im = cv2.imread(path_)
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im