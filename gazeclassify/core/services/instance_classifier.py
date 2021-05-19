# https://learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/
# https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py
import logging
import os
from dataclasses import dataclass
from typing import Union

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.services.analysis import Analysis
from gazeclassify.core.services.opencv_classifier import OpenCVClassifier
from gazeclassify.core.services.results import FrameResult


@dataclass
class VideoWriter:
    target_file: str

    def initiate(self, video_width: int, video_height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.result_video = cv2.VideoWriter(self.target_file, fourcc, 10, (video_width, video_height))

    def write(self, frame: np.ndarray) -> None:
        self.result_video.write(frame)

    def release(self) -> None:
        self.result_video.release()


@dataclass
class VideoReader:
    target_file: str

    @property
    def has_frame(self) -> Union[bool, np.ndarray]:
        return self._has_frame

    def initiate(self) -> None:
        self.capture = cv2.VideoCapture(self.target_file)

    def next_frame(self) -> np.ndarray:
        self._has_frame, frame = self.capture.read()
        return frame


@dataclass
class InstanceSegmentation:
    analysis: Analysis
    model_url: str = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
    proto_file_url: str = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt"

    def classify(self, name: str) -> None:
        writer = self._setup_video_writer(name)
        reader = self._setup_video_reader()

        for idx, record in enumerate(self.analysis.dataset.records):

            frame = reader.next_frame()

            if not reader.has_frame:
                logging.error("Video has ended prematurely")

            classifier = OpenCVClassifier(model_url=self.model_url, proto_file_url=self.proto_file_url)
            classifier.download_model()
            frameClone = classifier.classify_frame(frame)
            results = classifier.gaze_distance_to_instance(record)
            frame_results = FrameResult(idx, name, results)
            self.analysis.add_result(frame_results)

            writer.write(frameClone)

            idx += 1

            if idx == 2:
                print("DEBUG STOP")
                break

        writer.release()
        reader.capture.release()
        cv2.destroyAllWindows()

    def _setup_video_reader(self) -> VideoReader:
        source_file = str(self.analysis.dataset.world_video.file)
        reader = VideoReader(source_file)
        reader.initiate()
        return reader

    def _setup_video_writer(self, name: str) -> VideoWriter:
        video_target = os.path.expanduser("~/gazeclassify_data/") + f"{name}.mp4"
        logging.info(f"Writing export video to: {video_target}")
        world_video = self.analysis.dataset.world_video
        writer = VideoWriter(video_target)
        writer.initiate(world_video.width, world_video.height)
        return writer
