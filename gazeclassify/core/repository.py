from abc import ABC, abstractmethod
from typing import Dict, BinaryIO


class EyeTrackingDataRepository(ABC):

    @abstractmethod
    def load_gaze_data(self) -> Dict[str, BinaryIO]:
        ...

    @abstractmethod
    def load_video_metadata(self) -> Dict[str, int]:
        ...
