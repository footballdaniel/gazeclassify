from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.domain.dataset import NullDataset, Dataset
from gazeclassify.domain.results import FrameResult
from gazeclassify.domain.serialization import CSVSerializer, JsonSerializer
from gazeclassify.service.deletion import Deleter
from gazeclassify.service.stats import Results
from gazeclassify.thirdparty.moviepy_api import CompositeVideo


@dataclass
class Analysis:
    trial_name: str = "Trial"
    data_path: Path = Path.home().joinpath("gazeclassify_data")
    video_path: Path = data_path.joinpath("videos")
    model_path: Path = data_path.joinpath("models")
    result_path: Path = data_path.joinpath("results")
    results: List[FrameResult] = field(default_factory=list)
    recording: Dataset = NullDataset()

    def __post_init__(self) -> None:
        self.clear_data()

    def save_to_json(self) -> Analysis:
        logging.info(f"Writing results to: {str(self.result_path)}")
        serializer = JsonSerializer()
        Path.mkdir(self.result_path, parents=True, exist_ok=True)
        result_filename = self.result_path.joinpath("Result.json")
        serializer.encode(self.results, result_filename)
        return self

    def save_to_csv(self, folder: Optional[str] = None) -> Analysis:
        if folder is not None:
            self.result_path = Path(folder)
        logging.info(f"Writing results to: {str(self.result_path)}")
        serializer = CSVSerializer()
        Path.mkdir(self.result_path, parents=True, exist_ok=True)
        recording_path = Path(self.recording.metadata.recording_name).parent.parent.name
        file_name = recording_path + "_" + self.trial_name + ".csv"
        result_filename = self.result_path.joinpath(file_name)
        serializer.encode(self.results, result_filename)
        return self

    def export_video(self, folder: Optional[str] = None) -> Analysis:
        if folder is not None:
            self.result_path = Path(folder)
        video = CompositeVideo()
        video.index_videos(self.video_path)
        result_filename = self.result_path.joinpath(self.trial_name + "_composite.mp4")
        video.export(result_filename)
        return self

    def clear_data(self) -> Analysis:
        Deleter().clear_directory(self.result_path)
        Deleter().clear_directory(self.video_path)
        return self

    def add_result(self, result: FrameResult) -> None:
        self.results.append(result)

    def pieplot(self, filename: str = "pieplot.png") -> None:
        result_csv_file = str(self.result_path.joinpath(self.trial_name + ".csv"))
        (
            Results(result_csv_file)
                .piechart()
                .save(filename)
                .show()
        )
