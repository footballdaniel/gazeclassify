import logging
import os.path
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.domain.video import VideoWriter, VideoReader
from gazeclassify.services.analysis import Analysis
from gazeclassify.services.gaze_distance import DistanceToShape
from gazeclassify.services.model_loader import ClassifierLoader
from gazeclassify.services.results import Classification, FrameResult
from gazeclassify.thirdparty.opencv import OpenCVReader, OpenCVWriter
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


@dataclass
class SemanticSegmentation:
    analysis: Analysis
    video_filepath: Path = Path.home().joinpath("gazeclassify_data/videos")
    model_url: str = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"

    def _setup_video_reader(self, file: Path) -> VideoReader:
        reader = OpenCVReader(file)
        reader.initiate()
        return reader

    def _setup_video_writer(self, classifier_name: str) -> VideoWriter:
        Path.mkdir(self.video_filepath, parents=True, exist_ok=True)
        video_target = f"gazeclassify_data/videos/{classifier_name}.mp4"
        video_path = Path.home().joinpath(video_target)
        writer = OpenCVWriter(video_path)
        world_video = self.analysis.dataset.world_video
        writer.initiate(world_video.width, world_video.height)
        return writer

    def classify(self, classifier_name: str) -> None:
        writer = self._setup_video_writer(classifier_name)
        reader = self._setup_video_reader(self.analysis.dataset.world_video.file)

        model = self.download_model()

        segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        segment_image.load_model(model.file_path)
        target_classes = segment_image.select_target_classes(person=True)

        video_target = os.path.expanduser("~/gazeclassify_data/") + f"{classifier_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result_video = cv2.VideoWriter(video_target, fourcc, 10,
                                       (self.analysis.dataset.world_video.width,
                                        self.analysis.dataset.world_video.height))

        for idx, record in enumerate(self.analysis.dataset.records):

            logging.info(f"Instance segmentation at frame: {idx}")
            frame = reader.next_frame()

            if not reader.has_frame:
                logging.error("Video has ended prematurely")
                break

            classifier = PixellibTensorflowClassifier(self.model_url)
            segmask, output = segment_image.segmentFrame(frame, segment_target_classes=target_classes)

            # Only get the masks for all areas of type person
            people_masks = []
            person_class_id = 1
            for index in range(segmask["masks"].shape[-1]):
                if segmask['class_ids'][index] == person_class_id:
                    people_masks.append(segmask["masks"][:, :, index])

            # concatenate all masks (not distinguishing between codes)
            bool_concat = np.any(segmask["masks"], axis=-1)

            binary_image_mask = bool_concat.astype('uint8')
            pixel_distance = DistanceToShape(binary_image_mask)
            pixel_distance.detect_shape(positive_values=1)

            image_width = self.analysis.dataset.world_video.width
            image_height = self.analysis.dataset.world_video.height

            pixel_x = record.gaze.x * image_width
            pixel_y = image_height - (record.gaze.y * image_height)  # flip vertically
            print(f"GAZE: {pixel_x} + {pixel_y}")
            distance = pixel_distance.distance_2d(pixel_x, pixel_y)

            # Make alpha image to rgb
            int_concat = bool_concat.astype('uint8') * 255
            rgb_out = np.dstack([int_concat] * 3)


            # Visualize gaze overlay plot
            img_converted = Image.fromarray(output)
            import matplotlib.pyplot as plt  # type: ignore
            plt.imshow(img_converted)
            plt.scatter(pixel_x, pixel_y)
            # img_converted.show()
            # plt.show()

            idx += 1
            # Write video out
            result_video.write(rgb_out)

            classification = Classification(distance)
            frame_result = FrameResult(index, classifier_name, [classification])
            self.analysis.results.append(frame_result)


        result_video.release()
        reader.release()
        cv2.destroyAllWindows()

    def download_model(self) -> ClassifierLoader:
        model = ClassifierLoader(
            "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5",
            "gazeclassify_data/models")
        model.download_if_not_available()
        return model


@dataclass
class PixellibTensorflowClassifier:
    model_url: str = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"