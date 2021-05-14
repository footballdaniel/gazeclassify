import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Classification:
    name: str
    distances: List[Optional[float]]


class JsonSerializer:
    def encode(self, data: object, filename: str = "try.json") -> None:
        with open(filename, "w") as write_file:
            json.dump(
                data,
                write_file,
                indent=4,
                sort_keys=True,
                cls=ClassesToDictEncoder
            )


class ClassesToDictEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Dict[Any, Any]:
        return obj.__dict__
