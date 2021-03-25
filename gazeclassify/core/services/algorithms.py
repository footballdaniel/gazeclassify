from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO, Tuple, Dict, List


class Algorithm(ABC):

    @abstractmethod
    def analyze(self, frame: BinaryIO) -> Tuple[BinaryIO, Dict[str, str]]:
        ...


@dataclass
class PixelLibSegmentation:

    def analyze(self, frame: BinaryIO) -> None:
        pass


@dataclass
class ClassificationData:
    name: str
    result: List[int] = field(default_factory=list)
