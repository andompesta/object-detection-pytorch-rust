import torch
from abc import abstractmethod, ABCMeta
from typing import Optional
from os import path
from shutil import copyfile

from python.src.config import BaseConf
from python.src.utils import ensure_dir


class InitModule(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(InitModule, self).__init__()

    def init_weights(self) -> None:
        """Initialize weights if needed."""
        self.apply(self._init_weights_)

    @abstractmethod
    def _init_weights_(self, module: torch.nn.Module):
        """Child model has to define the initialization policy."""
        ...


class BaseModel(InitModule):
    def __init__(
            self,
            conf: BaseConf,
            **kw
    ):
        super(BaseModel, self).__init__()
        self.conf = conf
        self.name = conf.name

    def save(
            self,
            path_: str,
            is_best: Optional[bool] = False,
            file_name: str = 'checkpoint.pth.tar'
    ) -> None:
        if isinstance(self, torch.nn.DataParallel):
            state_dict = dict([(key, value.to("cpu")) for key, value in self.module.state_dict().items()])
        else:
            state_dict = dict([(key, value.to("cpu")) for key, value in self.state_dict().items()])

            torch.save(state_dict, ensure_dir(path.join(path_, file_name)))
            if is_best:
                copyfile(path.join(path_, file_name), path.join(path_, "model_best.pth.tar"))

    @classmethod
    def load(cls, conf: BaseConf, path_: str, mode: str = 'trained'):
        model = cls(conf)
        state_dict = torch.load(path_, map_location="cpu")

        if mode == 'pre-trained':
            strict = False
        elif mode == 'trained':
            state_dict = state_dict['state_dict']
            strict = True
        else:
            raise NotImplementedError()

        print(model.load_state_dict(state_dict, strict=strict))
        return model
