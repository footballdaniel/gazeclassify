from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List
import pathlib
import csv
import ffmpeg  # type: ignore
import numpy as np


@dataclass
class PupilDataDeserializer:
    _world_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        return self._world_timestamps

    @property
    def world_videoframes(self) -> Any:
        return self._world_videoframes

    def load_from_export_folder(self, path: str) -> PupilDataDeserializer:

        timestamps_file = self._get_world_timestamps_filepath(path)

        self._deserialize_world_timestamps(timestamps_file)

        self._deserialize_video(path)

        return self

    def _deserialize_world_timestamps(self, timestamps_file: pathlib.Path) -> None:
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

    def _read_lines(self, line_count: int, row: List[str]) -> None:
        if self.line_count == 0:
            self.line_count += 1
        else:
            self._world_timestamps.append(float(row[0]))
            self.line_count += 1

    def _get_world_timestamps_filepath(self, path: str) -> pathlib.Path:
        folder = pathlib.Path(path)
        full_filename = pathlib.Path.joinpath(folder, "world_timestamps.csv")
        return full_filename

    def _deserialize_video(self, path: str) -> None:
        folder = pathlib.Path(path)
        full_filename = pathlib.Path.joinpath(folder, "world.mp4")

        framebuffer = self._ffmpeg_decode(full_filename)

        height = 1088
        width = 1080
        ndarray = np.frombuffer(framebuffer, np.uint8).reshape([-1, height, width, 3])

        self._world_videoframes = ndarray

    def _ffmpeg_decode(self, full_filename: pathlib.Path) -> Any:

        replay_rate_input = 1
        log_level = 0

        frame_buffer, _ = (
            ffmpeg.input(full_filename, r=replay_rate_input)
            .output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", **{"loglevel": log_level}
            )
            .run(capture_stdout=True)
        )
        print(type(frame_buffer))
        return frame_buffer
