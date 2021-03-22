from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np  # type: ignore


@dataclass
class Metadata:
    recording_name: str
    world_video_width: int
    world_video_height: int
    world_video_framenumber: int


@dataclass
class Dataset(ABC):
    records: List[DataRecord]
    metadata: Metadata


@dataclass
class DataRecord:
    world_timestamp: float
    video: VideoFrame
    gaze: GazeData


@dataclass
class VideoFrame:
    width: int
    height: int
    frame: np.ndarray


@dataclass
class GazeData:
    x: float
    y: float
