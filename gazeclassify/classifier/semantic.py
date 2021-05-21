import logging
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore

from gazeclassify.domain.classification import Algorithm
from gazeclassify.domain.video import VideoReader, VideoWriter
from gazeclassify.service.analysis import Analysis
from gazeclassify.service.model_loader import ModelLoader
from gazeclassify.service.results import Classification, FrameResult
from gazeclassify.thirdparty.opencv import OpenCVReader, OpenCVWriter
from gazeclassify.thirdparty.pixellib import PixellibTensorflowClassifier


@dataclass
class SemanticSegmentation(Algorithm):
    _analysis: Analysis
    model_url: str = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def classify(self, classifier_name: str) -> None:
        writer = self._setup_video_writer(classifier_name)
        reader = self._setup_video_reader(self.analysis.dataset.world_video.file)
        model = self._download_model()

        for idx, record in enumerate(self.analysis.dataset.records):
            logging.info(f"Semantic segmentation at frame: {idx}")
            frame = reader.next_frame()

            classifier = PixellibTensorflowClassifier(model)
            classifier.set_target()
            frame = classifier.classify_frame(frame)
            result = classifier.gaze_distance_to_object(record)
            writer.write(frame)

            results = Classification(result)
            frame_result = FrameResult(idx, classifier_name, [results])
            self.analysis.results.append(frame_result)

        writer.release()
        reader.release()
        cv2.destroyAllWindows()

    def _download_model(self) -> ModelLoader:
        model = ModelLoader(
            "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5",
            str(self.analysis.model_path))
        model.download_if_not_available()
        return model

    def _setup_video_reader(self, file: Path) -> VideoReader:
        reader = OpenCVReader(file)
        reader.initiate()
        return reader

    def _setup_video_writer(self, classifier_name: str) -> VideoWriter:
        Path.mkdir(self.analysis.video_path, parents=True, exist_ok=True)
        video_target = Path(self.analysis.video_path).joinpath(f"{classifier_name}.mp4")
        writer = OpenCVWriter(video_target)
        world_video = self.analysis.dataset.world_video
        writer.initiate(world_video.width, world_video.height)
        return writer