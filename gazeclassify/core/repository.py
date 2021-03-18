from abc import ABC, abstractmethod

from gazeclassify.core.utils import Readable


class EyeTrackingDataRepository(ABC):

    @abstractmethod
    def load_capture(self, path: str) -> Readable:
        ...
