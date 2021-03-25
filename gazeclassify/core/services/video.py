from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class FrameReader(ABC):

    @abstractmethod
    def open(self, file: Path) -> None:
        ...

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

    @property
    @abstractmethod
    def file(self) -> Path:
        ...

    @abstractmethod
    def write_frame(self, frame: BinaryIO) -> None:
        ...

    @abstractmethod
    def release(self) -> None:
        ...
