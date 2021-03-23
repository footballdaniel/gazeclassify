from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Metadata:
    recording_name: str
    video_format: VideoMetadata


@dataclass
class VideoMetadata:
    file: Path
    width: int
    height: int
    frame_number: int
    frame_rate: int


@dataclass
class Dataset(ABC):
    records: List[DataRecord]
    metadata: Metadata


@dataclass
class DataRecord:
    world_timestamp: float
    video_frame_index: int
    gaze: GazeData


@dataclass
class GazeData:
    x: float
    y: float
