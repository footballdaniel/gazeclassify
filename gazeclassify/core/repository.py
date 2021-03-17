from dataclasses import dataclass
from abc import ABC, abstractmethod
from gazeclassify.core.model import Dataset


class EyeTrackingDataRepository(ABC):

    @abstractmethod
    def load_trial(self, path: str) -> Dataset:
        ...





