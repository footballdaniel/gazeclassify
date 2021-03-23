from abc import ABC, abstractmethod
from typing import Tuple, Dict, BinaryIO

from gazeclassify.core.model.dataset import Dataset
from gazeclassify.core.services.video import FrameReader


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, inputs: Dict[str, BinaryIO], video_metadata: Dict[str, int], video_capture: FrameReader) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


