from python.src.config import ResNet18Conf, FPN18Conf
from python.src.models import BaseModel, ResNet18, RegionProposalNetwork, FPN

class FasterRCNN(BaseModel):
    def __init__(self, conf):
        super(FasterRCNN, self).__init__(conf)

        if isinstance(conf.backbone, ResNet18Conf):
            self.backbone = ResNet18(conf.backbone)
        elif isinstance(conf.baclbone, FPN18Conf):
            self.backbone = FPN(conf.backbone)
        else:
            raise NotImplementedError(f"Backbone type {conf.backbone.name} not yet implemented")

        self.rpn = RegionProposalNetwork(conf.rpn, self.backbone.output_shapes())

        self.roi_heads = ...

