# https://learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/
# https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.services.analysis import Analysis
from gazeclassify.services.opencv_classifier import OpenCVClassifier
from gazeclassify.services.results import FrameResult
from gazeclassify.thirdparty.opencv import OpenCVWriter, OpenCVReader



@dataclass
class InstanceSegmentation:
    analysis: Analysis
    model_url: str = "https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel"
    proto_file_url: str = "https://raw.githubusercontent.com/spmallick/learnopencv/master/OpenPose-Multi-Person/pose/coco/pose_deploy_linevec.prototxt"

    def classify(self, classifier_name: str) -> None:
        writer = self._setup_video_writer(classifier_name)
        reader = self._setup_video_reader(self.analysis.dataset.world_video.file)

        for idx, record in enumerate(self.analysis.dataset.records):
            logging.info(f"Instance segmentation at frame: {idx}")
            frame = reader.next_frame()

            if not reader.has_frame:
                logging.error("Video has ended prematurely")

            classifier = OpenCVClassifier(model_url=self.model_url, proto_file_url=self.proto_file_url)
            classifier.download_model()
            frameClone = classifier.classify_frame(frame)
            results = classifier.gaze_distance_to_instance(record)
            frame_results = FrameResult(idx, classifier_name, results)
            self.analysis.add_result(frame_results)
            writer.write(frameClone)
            idx += 1

        writer.release()
        reader.release()
        cv2.destroyAllWindows()

    def _setup_video_reader(self, file: Path) -> OpenCVReader:
        reader = OpenCVReader(file)
        reader.initiate()
        return reader

    def _setup_video_writer(self, classifier_name: str) -> OpenCVWriter:
        video_target = f"gazeclassify_data/video/{classifier_name}.mp4"
        video_path = Path.home().joinpath(video_target)
        writer = OpenCVWriter(video_path)
        world_video = self.analysis.dataset.world_video
        writer.initiate(world_video.width, world_video.height)
        return writer
