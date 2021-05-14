import os.path
from dataclasses import dataclass, field
from typing import List

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.model.dataset import NullDataset, Dataset
from gazeclassify.core.services.results import Classification, JsonSerializer, FrameResults


@dataclass
class Analysis:
    data_path: str = os.path.expanduser("~/gazeclassify_data/")
    results: List[FrameResults] = field(default_factory=list)
    dataset: Dataset = NullDataset()

    def save_to_json(self) -> None:
        serializer = JsonSerializer()
        serializer.encode(self.results, f"{self.dataset.metadata.recording_name}.json")

    def clear_data(self) -> None:
        raise NotImplementedError
