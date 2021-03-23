from abc import ABC, abstractmethod
from typing import BinaryIO


class FrameReader(ABC):

    @property
    @abstractmethod
    def current_frame_index(self) -> int:
        ...

    @abstractmethod
    def get_frame(self) -> BinaryIO:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class FrameWriter(ABC):

    @property
    @abstractmethod
    def current_frame_index(self) -> int:
        ...

    @abstractmethod
    def write_frame(self, frame: BinaryIO) -> None:
        ...

    def release(self) -> None:
        ...
