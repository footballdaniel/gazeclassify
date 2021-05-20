from abc import ABC, abstractmethod
from typing import Union

import numpy as np  # type: ignore


class VideoWriter(ABC):

    @abstractmethod
    def initiate(self, video_width: int, video_height: int) -> None:
        ...

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class VideoReader(ABC):

    @property
    @abstractmethod
    def has_frame(self) -> Union[bool, np.ndarray]:
        ...

    @abstractmethod
    def initiate(self) -> None:
        ...

    @abstractmethod
    def next_frame(self) -> np.ndarray:
        ...

    @abstractmethod
    def release(self) -> None:
       ...