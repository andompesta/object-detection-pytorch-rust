from typing import List

class MatcherConf(object):
    def __init__(
            self,
            thresholds: List[float],
            labels: List[int],
            allow_low_quality_matches: bool = True
    ):

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

class ProposalMatcherConf(MatcherConf):
    def __init__(
            self,
            roi_thresholds=[0.5],
            roi_labels=[0, 1],
            allow_low_quality_matches=False
    ):
        super(ProposalMatcherConf, self).__init__(
            thresholds=roi_thresholds,
            labels=roi_labels,
            allow_low_quality_matches=allow_low_quality_matches
        )