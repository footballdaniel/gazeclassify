from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from gazeclassify.domain.results import FrameResult, Classification, InstanceClassification


@dataclass
class ClassificationBuilder:
    distance: Optional[float] = None
    joint: str = ""
    person_id: int = 0

    def with_distance(self, distance: Optional[float]) -> ClassificationBuilder:
        self.distance = distance
        return self

    def with_joint(self, joint: str) -> ClassificationBuilder:
        self.joint = joint
        return self

    def with_person_id(self, person_id: int) -> ClassificationBuilder:
        self.person_id = person_id
        return self

    def build(self) -> Classification:
        classification = Classification(
            self.distance
        )
        return classification


@dataclass
class InstanceClassificationBuilder(ClassificationBuilder):

    def build(self) -> InstanceClassification:
        classification = InstanceClassification(
            self.distance,
            self.joint,
            self.person_id
        )
        return classification


@dataclass
class FrameResultBuilder:
    frame_id: int = 0
    name: str = ""
    classifications: List[Classification] = field(default_factory=list)

    def with_frame_id(self, frame_id: int) -> FrameResultBuilder:
        self.frame_id = frame_id
        return self

    def with_name(self, name: str) -> FrameResultBuilder:
        self.name = name
        return self

    def with_classification(self, classification: Classification) -> FrameResultBuilder:
        self.classifications.append(classification)
        return self

    def build(self) -> FrameResult:
        result = FrameResult(
            self.frame_id,
            self.name,
            self.classifications
        )
        return result
