from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.domain.video import VideoWriter, VideoReader


@dataclass
class OpenCVWriter(VideoWriter):
    target_file: Path

    def initiate(self, video_width: int, video_height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.result_video = cv2.VideoWriter(str(self.target_file), fourcc, 10, (video_width, video_height))

    def write(self, frame: np.ndarray) -> None:
        self.result_video.write(frame)

    def release(self) -> None:
        self.result_video.release()


@dataclass
class OpenCVReader(VideoReader):
    target_file: Path

    @property
    def has_frame(self) -> bool:
        if self._has_frame == True:
            return True
        else:
            return False

    def initiate(self) -> None:
        self.capture = cv2.VideoCapture(str(self.target_file))

    def next_frame(self) -> np.ndarray:
        self._has_frame, frame = self.capture.read()
        return frame

    def release(self) -> None:
        self.capture.release()