from abc import ABC, abstractmethod
from typing import BinaryIO


class FrameSeeker(ABC):

    @property
    @abstractmethod
    def current_frame_index(self) -> int:
        ...

    @abstractmethod
    def get_frame(self) -> BinaryIO:
        ...