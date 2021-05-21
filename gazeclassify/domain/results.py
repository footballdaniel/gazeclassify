import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class Classification:
    distance: Optional[float]


@dataclass
class InstanceClassification(Classification):
    joint: str
    person_id: int


@dataclass
class FrameResult:
    frame_id: int
    name: str
    classifications: List[Classification] = field(default_factory=list)


class CSVSerializer:
    def encode(self, data: List[FrameResult], filename: Path) -> None:
        print("NOT IMPLEMNTEND CSV Serialization")
        pass


class JsonSerializer:
    def encode(self, data: object, filename: Path) -> None:
        with open(str(filename), "w") as write_file:
            json.dump(
                data,
                write_file,
                indent=4,
                sort_keys=False,
                cls=ClassesToDictEncoder
            )


class ClassesToDictEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Dict[Any, Any]:
        return obj.__dict__
