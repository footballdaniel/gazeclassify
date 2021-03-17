from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np  # type: ignore


@dataclass
class Metadata:
    recording_name: str


@dataclass
class Dataset:
    records: List[DataRecord]
    metadata: Metadata


@dataclass
class DataRecord:
    world_timestamp: float
    world_frame: VideoFrame
    gaze: GazeData


@dataclass
class VideoFrame:
    width: int
    height: int
    frame: np.ndarray


@dataclass
class GazeData:
    x: int
    y: int
