from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import pathlib
import csv


@dataclass
class PupilDataDeserializer:
    _world_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        return self._world_timestamps

    def load_from_export_folder(self, path: str) -> PupilDataDeserializer:

        timestamps_file = self._get_world_timestamps_filepath(path)

        self._deserialize_world_timestamps(timestamps_file)

        self._deserialize_video(path)

        return self

    def _deserialize_world_timestamps(self, timestamps_file):
        if timestamps_file.exists():
            with open(timestamps_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                self.line_count = 0
                for row in csv_reader:
                    self._read_lines(self.line_count, row)
        else:
            raise FileNotFoundError(
                "Could not find the file world_timestamps.csv in folder"
            )

    def _read_lines(self, line_count: int, row: List[str]):
        if self.line_count == 0:
            self.line_count += 1
        else:
            self._world_timestamps.append(float(row[0]))
            self.line_count += 1

    def _get_world_timestamps_filepath(self, path: str) -> pathlib.Path:
        folder = pathlib.Path(path)
        world_timestamps = pathlib.Path.joinpath(folder, "world_timestamps.csv")
        return world_timestamps

    def _deserialize_video(self, path: str):
        pass


PupilDataDeserializer().load_from_export_folder("gazepy/tests/data/")
