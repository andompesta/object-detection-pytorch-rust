import copy

from typing import List

from python.src.utils import ensure_dir, save_data_to_json, load_data_from_json
from abc import ABC


class AnchorMatcherConf(object):
    def __init__(
            self,
            iou_thresholds: List[float] = [0.3, 0.7],
            iou_labels: List[int] = [0, -1, 1],
            allow_low_quality_matches: bool = True
    ):
        """
        :param iou_thresholds: a list of Intersection-over-Union thresholds used to stratify predictions into levels.
        :param iou_labels: a list of Intersection-over-Union label predictions belonging at each level. A label can be
            one of {-1, 0, 1} signifying {ignore, negative class, positive class}, respectively.
        :param allow_low_quality_matches: if True, produce additional matches for predictions with maximum match quality
            lower than high_threshold.
            See set_low_quality_matches_ for more details.

        For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.

                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """

        self.iou_thresholds = iou_thresholds
        self.iou_labels = iou_labels
        self.allow_low_quality_matches = allow_low_quality_matches

class BaseConf(ABC):
    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        return output

    def save(self, path_: str):
        save_data_to_json(ensure_dir(path_), self.to_dict())

    @classmethod
    def from_dict(cls, json_object: dict) -> object:
        return cls(**json_object)

    @classmethod
    def load(cls, path_: str):
        json_obj = load_data_from_json(path_)
        return cls(**dict(json_obj))