from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import ffmpeg  # type: ignore
import numpy as np  # type: ignore


@dataclass
class PupilDataLoader:
    _foldername: str = "Video_000"
    _world_video_height: int = 0
    _world_video_width: int = 0
    _world_video_framenumber: int = 0
    _world_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        return self._world_timestamps

    @property
    def world_videoframes(self) -> np.ndarray:
        return self._world_videoframes

    @property
    def world_video_height(self) -> int:
        return self._world_video_height

    @property
    def world_video_width(self) -> int:
        return self._world_video_width

    @property
    def world_video_framenumber(self) -> int:
        return self._world_video_framenumber

    @property
    def foldername(self) -> str:
        return self._foldername

    def load_from_export_folder(self, path: str, default_video_name: str = "world.mp4") -> PupilDataLoader:
        timestamps_file = self._get_world_timestamps_filepath(path)
        self._get_folder_name(path)
        self._deserialize_world_timestamps(timestamps_file)
        self._deserialize_video(path, default_video_name)
        return self

    def _get_folder_name(self, path: str) -> None:
        path_to_folder = Path(path)
        self._foldername = path_to_folder.stem

    def _get_world_timestamps_filepath(self, path: str) -> Path:
        folder = Path(path)
        full_filename = Path.joinpath(folder, "world_timestamps.csv")
        return full_filename

    def _deserialize_world_timestamps(self, timestamps_file: Path) -> None:
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

    def _deserialize_video(self, path: str, default_video_name: str) -> None:
        folder = Path(path)
        full_filename = Path.joinpath(folder, default_video_name)
        self._ffmpeg_decode_size(full_filename)
        framebuffer = self._ffmpeg_decode(full_filename)
        frames = np.frombuffer(framebuffer, np.uint8).reshape([-1, self._world_video_width, self.world_video_height, 3])

        self._world_videoframes = frames

    def _ffmpeg_decode_size(self, full_filename: Path) -> None:
        probe = ffmpeg.probe(full_filename)
        video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video['width'])  # type: ignore
        height = int(video['height'])  # type: ignore

        self._world_video_height = height
        self._world_video_width = width

    def _ffmpeg_decode(self, full_filename: Path) -> Any:
        replay_rate_input = 1
        log_level = 0
        frame_buffer, _ = (
            ffmpeg.input(full_filename, r=replay_rate_input)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", **{"loglevel": log_level})
                .run(capture_stdout=True)
        )
        print(type(frame_buffer))
        return frame_buffer
