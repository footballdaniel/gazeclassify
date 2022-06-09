from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Classification:
    distance: Optional[float]

@dataclass
class InstanceClassification(Classification):
    name: str
    id: int


@dataclass
class FrameResult:
    frame_id: int
    name: str
    classifications: List[Classification] = field(default_factory=list)
