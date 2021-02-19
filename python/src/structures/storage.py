from numpy import ndarray
from torch import Tensor
from collections import defaultdict

from typing import *
from fvcore.common.history_buffer import HistoryBuffer

class Logs:
    """
    singleton wrapper around history buffer
    """
    _instance_ = None

    @staticmethod
    def get_instance():
        """
        class access method
        :return:
        """
        if Logs._instance_ == None:
            Logs()

        return Logs._instance_

    def __init__(
            self,
            start_iter: int = 0,
            max_length: int = 1000000
    ):
        """
        private constructor
        """
        if Logs._instance_ != None:
            raise Warning("This class is a singleton!")
        else:
            self._iter = start_iter
            self._history = defaultdict(HistoryBuffer(max_length))
            self._vis_data = []
            self._histograms = []
            self._smoothing_hints = {}
            self._latest_scalars = {}
            Logs._instance_ = self

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []

    def clear_history(self):
        self._history = defaultdict(HistoryBuffer(self._history._max_length))

    @property
    def iter(self):
        """
        :return: The current iteration number. When used together with a trainer, this is ensured to be the same
        as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)


    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(
            self,
            window_size=20
    ):
        """
        Similar to :meth:`latest`, but the returned values are either the un-smoothed original latest value,
        or a median of the given window_size, depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        :param window_size: temporal windows size used to compute the average
        :return:
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either:
            (1) Call this function to increment storage.iter when needed. Or
            (2) Set `storage.iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1



    def put_image(
            self,
            img_name: str,
            img_tensor: Union[ndarray, Tensor]
    ):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on tensorboard.
        :param img_name: The name of the image to put into tensorboard.
        :param img_tensor: An `uint8` or `float` Tensor of shape `[channel, height, width]` where `channel` is 3. The
            image format should be RGB. The elements in img_tensor can either have values in [0, 1] (float32) or
            [0, 255] (uint8). The `img_tensor` will be visualized in tensorboard.
        :return:
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(
            self,
            name: str,
            value: Any,
            smoothing_hint: bool = True
    ):
        """

        :param name: name of the scalar
        :param value: value of the scalar to log
        :param smoothing_hint: a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal
        :return:
        """
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(
            self,
            smoothing_hint=True,
            **kwargs
    ):
        """
        Put multiple scalars from keyword arguments.
        Examples:
            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)
