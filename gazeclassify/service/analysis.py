from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.domain.dataset import NullDataset, Dataset
from gazeclassify.service.deletion import Deleter
from gazeclassify.domain.results import JsonSerializer, FrameResult, CSVSerializer

import logging

@dataclass
class Analysis:
    data_path: Path = Path.home().joinpath("gazeclassify_data")
    video_path: Path = data_path.joinpath("videos")
    model_path: Path = data_path.joinpath("models")
    result_path: Path = data_path.joinpath("results")
    results: List[FrameResult] = field(default_factory=list)
    dataset: Dataset = NullDataset()


    def save_to_json(self) -> Analysis:
        serializer = JsonSerializer()
        Path.mkdir(self.result_path, parents=True, exist_ok=True)
        result_filename = self.result_path.joinpath(f"{self.dataset.metadata.recording_name}.json")
        serializer.encode(self.results, result_filename)
        return self

    def save_to_csv(self) -> Analysis:
        serializer = CSVSerializer()
        Path.mkdir(self.result_path, parents=True, exist_ok=True)
        result_filename = self.result_path.joinpath(f"{self.dataset.metadata.recording_name}.json")
        serializer.encode(self.results, result_filename)
        return self

    def clear_data(self) -> Analysis:
        Deleter().clear_directory(self.result_path)
        Deleter().clear_directory(self.video_path)
        return self

    def add_result(self, result: FrameResult) -> None:
        self.results.append(result)
