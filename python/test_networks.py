import torch
import os
from dynaconf import settings
from typing import Dict

from python.src.config import ResNet18Conf, FPN18Conf
from python.src.models import ResNet18, FPN
from python.src.preprocessing.classification import preprocess_image_imagenet

def get_idx_to_labels(path_: str) -> Dict[int, str]:
    with open(path_, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return dict([(idx, c) for idx, c in enumerate(categories)])

if __name__ == '__main__':

    # conf = ResNet18Conf(num_classes=1000)
    # model = ResNet18.load(
    #     conf,
    #     os.path.join(
    #         settings.get("ckp_dir"),
    #         "import",
    #         "restnet-18.pth.tar"
    #     ),
    #     mode="pre-trained"
    # )
    conf = FPN18Conf()
    model = FPN(conf)
    model.eval()
    print(model.output_shapes())

    x = preprocess_image_imagenet(
        os.path.join(
            settings.get("data_dir"),
            "imagenet",
            "dog.jpg"
        )
    )

    IDX_TO_LABELS = get_idx_to_labels(os.path.join(
        settings.get("data_dir"),
        "imagenet",
        "imagenet_classes.txt"
    ))

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)

    top5_prob, top5_catid = torch.topk(prob, 5)
    for idx, (p, c) in enumerate(zip(top5_prob.squeeze(), top5_catid.squeeze())):
        print(IDX_TO_LABELS[c.item()], p.item())

