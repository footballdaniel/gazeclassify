from abc import ABC, abstractmethod
from typing import Tuple, Dict

from gazeclassify.core.model import Dataset
from gazeclassify.core.utils import Readable


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, inputs: Dict[str, Readable]) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError
