from abc import ABC, abstractmethod
from typing import Tuple, Dict

from gazeclassify.core.model import Dataset
from gazeclassify.core.utils import Readable
from gazeclassify.core.video import FrameSeeker


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, inputs: Dict[str, Readable], video_metadata: Dict[str, int], video_capture: FrameSeeker) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


