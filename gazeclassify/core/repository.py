from abc import ABC, abstractmethod
from typing import Dict

from gazeclassify.core.video import FrameSeeker
from gazeclassify.core.utils import Readable


class EyeTrackingDataRepository(ABC):

    @abstractmethod
    def load_gaze_data(self) -> Readable:
        ...

    @abstractmethod
    def load_video_capture(self) -> FrameSeeker:
        ...

    @abstractmethod
    def load_video_metadata(self) -> Dict[str, int]:
        ...