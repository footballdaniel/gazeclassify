from abc import ABC, abstractmethod
from typing import Dict, BinaryIO

from gazeclassify.core.video import FrameSeeker

class EyeTrackingDataRepository(ABC):

    @abstractmethod
    def load_gaze_data(self) -> Dict[str, BinaryIO]:
        ...

    @abstractmethod
    def load_video_capture(self) -> FrameSeeker:
        ...

    @abstractmethod
    def load_video_metadata(self) -> Dict[str, int]:
        ...