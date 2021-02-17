import torch

from python.src.config import RegionProposalNetworkConf
from python.src.models.components import DefaultAnchorGenerator, Matcher

class RegionProposalNetwork(torch.nn.Module):
    def __init__(
            self,
            conf: RegionProposalNetworkConf
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = DefaultAnchorGenerator(conf.anchor_generator)
        self.anchor_matcher = Matcher(conf.anchor_matcher)




