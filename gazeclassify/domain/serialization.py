import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, BinaryIO, List, Any, cast

from gazeclassify.domain.dataset import Dataset
from gazeclassify.domain.results import FrameResult, InstanceClassification


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, gaze_data: Dict[str, BinaryIO], video_metadata: Dict[str, str]) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError
