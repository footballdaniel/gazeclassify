from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union


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
    classifications: List[ClassificationData] = field(default_factory=list)

@dataclass
class ClassificationData:
    classification: Dict[str, Union[List[int], List[Instance]]]

@dataclass
class Instance:
    name: str
    pixel_location: Position

@dataclass
class Position:
    x: int
    y: int

@dataclass
class DataRecord:
    world_timestamp: float
    video_frame_index: int
    gaze: GazeData

@dataclass
class GazeData:
    x: float
    y: float
