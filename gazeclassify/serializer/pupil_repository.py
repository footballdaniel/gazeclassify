import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, BinaryIO

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.repository import EyeTrackingDataRepository
from gazeclassify.core.utils import Readable
from gazeclassify.core.video import FrameSeeker
from gazeclassify.serializer.pupil_data_loader import PupilDataLoader


@dataclass
class OpenCVFrameSeeker(FrameSeeker):
    folder_path: str
    _current_frame_index: int = -1

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index if self._current_frame_index >= 0 else 0

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
class PupilInvisibleRepository(EyeTrackingDataRepository):
    folder_path: str

    def load_gaze_data(self) -> Dict[str, Readable]:
        data = PupilDataLoader().load_from_export_folder(self.folder_path)
        data_dict = {
            "world timestamps": data.world_timestamps,
            "world video height": data.world_video_height,
            "world video width": data.world_video_width,
            "world video framenumber": data.world_video_framenumber,
            "folder name": data.foldername,
            "gaze x": data.gaze_x,
            "gaze y": data.gaze_y,
            "gaze timestamps": data.gaze_timestamps,
            "world video frames": data.world_videoframes
        }
        return data_dict

    def load_video_capture(self) -> FrameSeeker:
        frame_seeker = OpenCVFrameSeeker(self.folder_path)
        frame_seeker.open_capture()
        return frame_seeker

    def load_video_metadata(self) -> Dict[str, int]:
        video_file = self._get_file_name(self.folder_path)
        capture = cv2.VideoCapture(video_file)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
        frame_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        video_metadata = {
            'width': width,
            'height': height,
            'frame rate': frame_rate,
            'frame number': frame_number
        }
        return video_metadata

    def _get_file_name(self, path: str) -> str:
        filename = path + "/world.mp4"
        return filename
