import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Classification:
    distance: float


@dataclass
class InstanceClassification(Classification):
    joint: str
    person_id: int


@dataclass
class FrameResults:
    frame_id: int
    name: str
    classifications: List[Classification] = field(default_factory=list)


class JsonSerializer:
    def encode(self, data: object, filename: str = "try.json") -> None:
        with open(filename, "w") as write_file:
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

