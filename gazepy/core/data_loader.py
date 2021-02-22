from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import pathlib


@dataclass
class PupilDataLoader:
    _world_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        return self._world_timestamps

    def load_from_export_folder(self, path: str) -> PupilDataLoader:

        world_timestamps = self._get_world_timestamps_filepath(path)

        if world_timestamps.exists():
            self._world_timestamps.append(2.955558)

        else:
            raise FileNotFoundError(
                "Could not find the file world_timestamps.csv in folder"
            )

        return self

    def _get_world_timestamps_filepath(self, path: str) -> pathlib.Path:
        folder = pathlib.Path(path)
        world_timestamps = pathlib.Path.joinpath(folder, "world_timestamps.csv")
        return world_timestamps
