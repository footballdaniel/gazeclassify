import io
from dataclasses import dataclass
from typing import BinaryIO, Any, cast

import cv2  # type: ignore

from gazeclassify.core.services.video import FrameReader, FrameWriter


@dataclass
class OpenCVFrameReader(FrameReader):
    folder_path: str
    _current_frame_index: int = -1

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index if self._current_frame_index >= 0 else 0

    def release(self) -> None:
        self._capture.release()

    def _get_file_name(self, path: str) -> str:
        return path + "/world.mp4"

    def open_capture(self) -> None:
        file_name = self._get_file_name(self.folder_path)
        self._capture = cv2.VideoCapture(file_name)

    def get_frame(self) -> BinaryIO:
        has_frame, frame = self._capture.read()
        if not has_frame:
            self._capture.release()
        bytesio = self.convert_to_bytesio(frame)
        self._current_frame_index += 1
        return bytesio

    def convert_to_bytesio(self, frame: bytes) -> BinaryIO:
        bytestream = cv2.imencode('.jpg', frame)[1].tobytes()
        bytesio = io.BytesIO(bytestream)
        return bytesio


@dataclass
class OpenCVFrameWriter(FrameWriter):
    output_video_name: str = ""
    codec: str = 'DIVX'
    frames_per_second: int = 25
    width: int = 100
    height: int = 100
    _current_frame_index: int = 0
    _capture: Any = None

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    def write_frame(self, frame: BinaryIO) -> None:
        if self._capture == None:
            self._create_stream()

        frame = self._resize(frame)

        self._write(frame)

    def _create_stream(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._capture = cv2.VideoWriter(
            self.output_video_name,
            fourcc,
            self.frames_per_second,
            (self.width, self.height)
        )

    def _resize(self, frame: BinaryIO) -> BinaryIO:
        cv2frame: Any = cast(Any, frame)
        resized = cv2.resize(cv2frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized_frame: BinaryIO = cast(BinaryIO, resized)
        return resized_frame

    def _write(self, frame: BinaryIO) -> None:
        self._capture.write(frame)

    def release(self) -> None:
        self._capture.release()
