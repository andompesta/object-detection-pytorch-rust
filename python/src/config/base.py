import copy
from python.src.utils import ensure_dir, save_data_to_json, load_data_from_json
from abc import ABC

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