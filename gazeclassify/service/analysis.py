from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.domain.dataset import NullDataset, Dataset
from gazeclassify.service.deletion import FileDeleter
from gazeclassify.service.results import JsonSerializer, FrameResult

import logging

@dataclass
class Analysis:
    data_path: Path = Path.home().joinpath("gazeclassify_data")
    results: List[FrameResult] = field(default_factory=list)
    dataset: Dataset = NullDataset()

    def save_to_json(self) -> None:
        serializer = JsonSerializer()
        serializer.encode(self.results, f"{self.dataset.metadata.recording_name}.json")

    def clear_data(self) -> None:
        FileDeleter().clear_files(self.data_path, "*.mp4")
        FileDeleter().clear_files(self.data_path, "*.json")

    def set_logger(self, level: Union[int, str]) -> None:
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    def add_result(self, result: FrameResult) -> None:
        self.results.append(result)
