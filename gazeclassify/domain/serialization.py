from abc import ABC, abstractmethod
from typing import Tuple, Dict, BinaryIO

from gazeclassify.domain.dataset import Dataset


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, gaze_data: Dict[str, BinaryIO], video_metadata: Dict[str, str]) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError
