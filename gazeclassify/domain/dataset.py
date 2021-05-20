from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Metadata:
    recording_name: str


@dataclass
class Segmentation:
    name: str
    gaze_distance: float


@dataclass
class VideoData:
    file: Path
    width: int
    height: int
    frame_number: int
    frame_rate: int


@dataclass
class Dataset:
    records: List[DataRecord]
    metadata: Metadata
    world_video: VideoData


@dataclass
class Instance:
    name: str
    pixel_location: Position


@dataclass
class Position:
    x: int
    y: int


@dataclass
class GazeData:
    x: float
    y: float


@dataclass
class DataRecord:
    world_timestamp: float
    video_frame_index: int
    gaze: GazeData


@dataclass
class NullRecord(DataRecord):
    world_timestamp: float = 0.
    video_frame_index: int = 0
    gaze: GazeData = GazeData(0., 0.)


@dataclass
class NullDataset(Dataset):
    records: List[DataRecord] = field(default_factory=list)
    metadata: Metadata = Metadata("")
    world_video: VideoData = VideoData(Path(""), 0, 0, 0, 0)
