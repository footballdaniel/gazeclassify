from abc import ABC, abstractmethod
from typing import Tuple, Dict, BinaryIO

from gazeclassify.core.model import Dataset
from gazeclassify.core.video import FrameSeeker


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, inputs: Dict[str, BinaryIO], video_metadata: Dict[str, int], video_capture: FrameSeeker) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


