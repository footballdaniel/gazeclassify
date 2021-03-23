from dataclasses import dataclass
from typing import Dict, BinaryIO

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.repository import EyeTrackingDataRepository
from gazeclassify.core.services.video import FrameReader
from gazeclassify.serializer.pupil_data_loader import PupilDataLoader
from gazeclassify.thirdparty.opencv import OpenCVFrameReader


@dataclass
class PupilInvisibleRepository(EyeTrackingDataRepository):
    folder_path: str

    def load_gaze_data(self) -> Dict[str, BinaryIO]:
        loader = PupilDataLoader()

        gaze_filestream = loader.access_gaze_file(self.folder_path)
        world_timestamps_filestream = loader.access_world_timestamps(self.folder_path)

        filestream_dict = {
            "gaze location": gaze_filestream,
            "world timestamps": world_timestamps_filestream
        }
        return filestream_dict

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
