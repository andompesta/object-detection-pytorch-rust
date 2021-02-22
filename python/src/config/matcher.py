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