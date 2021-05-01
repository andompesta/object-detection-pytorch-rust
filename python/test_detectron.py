from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import os
from dynaconf import settings

from src.utils import show_image
from src.preprocessing.classification import preprocess_image_imagenet

import torch
from einops import rearrange

if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

    cfg.MODEL.WEIGHaTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    fpn_backbone = predictor.model.backbone

    print("pippo")


    im = preprocess_image_imagenet(
        os.path.join(
            settings.get("data_dir"),
            "coco",
            "input.jpg"
        )
    )
    with torch.no_grad():

        res = fpn_backbone.forward(
            im.to("cuda")
        )
    outputs = predictor(im)

    print()
