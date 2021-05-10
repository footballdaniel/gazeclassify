import json
import logging
import os.path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import cv2  # type: ignore
import numpy as np  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.model.dataset import NullDataset, Dataset
from gazeclassify.core.services.gaze_distance import PixelDistance
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilInvisibleSerializer
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


@dataclass
class ModelLoader:
    data_path: str = "~/gazeclassify_data/"

    def _data_path_in_home_directory(self) -> str:
        return os.path.expanduser(self.data_path)

    def download_if_not_available(self, model_file: str = "mask_rcnn_coco.h5") -> None:
        file_path = self._data_path_in_home_directory() + model_file
        if not os.path.exists(file_path):
            self._download_file(model_file)
        self.path = file_path

    def _download_file(self, model_file: str) -> None:
        print("FILE NOT EXISTING")
        print(self.data_path + model_file)
        from urllib import request
        remote_url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"
        local_file = self._data_path_in_home_directory() + 'mask_rcnn_coco.h5'
        request.urlretrieve(remote_url, local_file)


@dataclass
class Classification:
    name: str
    distance_from_gaze: float


@dataclass
class FrameResult:
    records: List[Classification]


class ClassesToDictEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Dict[Any, Any]:
        return obj.__dict__


class JsonSerializer:
    def encode(self, data: object, filename: str = "try.json") -> None:
        with open(filename, "w") as write_file:
            json.dump(
                data,
                write_file,
                indent=4,
                sort_keys=True,
                cls=ClassesToDictEncoder
            )


@dataclass
class Analysis:
    data_path: str = os.path.expanduser("~/gazeclassify_data/")
    results: List[FrameResult] = field(default_factory=list)
    dataset: Dataset = NullDataset()


    def save_to_json(self) -> None:
        serializer = JsonSerializer()
        serializer.encode(self.results, "test.json")


@dataclass
class PupilInvisibleLoader:
    analysis: Analysis

    def from_trial_folder(self, path: str) -> None:
        file_repository = PupilInvisibleRepository(path)
        gaze_data = file_repository.load_gaze_data()
        video_metadata = file_repository.load_video_metadata()
        serializer = PupilInvisibleSerializer()
        self.analysis.dataset = serializer.deserialize(gaze_data, video_metadata)

@dataclass
class SemanticSegmentation:
    analysis: Analysis

    def classify(self, name: str) -> None:

        source_file = str(self.analysis.dataset.world_video.file)

        logger = logging.getLogger(__name__)
        logger.setLevel('INFO')

        model = ModelLoader("~/gazeclassify_data/")
        model.download_if_not_available("mask_rcnn_coco.h5")

        segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        segment_image.load_model(model.path)
        target_classes = segment_image.select_target_classes(person=True)

        # SEND FRAME TO WRITER
        video_target = os.path.expanduser(f"~/gazeclassify_data/{name}.avi")
        result_video = cv2.VideoWriter(video_target, cv2.VideoWriter_fourcc(*'MP4V'), 10,
                                       (self.analysis.dataset.world_video.width,
                                        self.analysis.dataset.world_video.height))

        capture = cv2.VideoCapture(source_file)

        index = 0
        for record in self.analysis.dataset.records:

            hasframe, frame = capture.read()

            if not hasframe:
                print("Ran out of frames")

            segmask, output = segment_image.segmentFrame(frame, segment_target_classes=target_classes)

            # img_converted = Image.fromarray(output)
            # img_converted.show()

            # Only get the masks for all areas of type person
            people_masks = []
            person_class_id = 1
            for index in range(segmask["masks"].shape[-1]):
                if segmask['class_ids'][index] == person_class_id:
                    people_masks.append(segmask["masks"][:, :, index])

            # concatenate all masks (not distinguishing between codes)
            bool_concat = np.any(segmask["masks"], axis=-1)

            binary_image_mask = bool_concat.astype('uint8')
            pixel_distance = PixelDistance(binary_image_mask)
            pixel_distance.detect_shape(positive_values=1)

            print(f"GAZE: {record.gaze.x} + {record.gaze.y}")
            distance = pixel_distance.distance_gaze_to_shape(record.gaze.x, record.gaze.y)

            # Make alpha image to rgb
            int_concat = bool_concat.astype('uint8') * 255
            rgb_out = np.dstack([int_concat] * 3)

            # Write video out
            result_video.write(rgb_out)

            # Append results
            classification = Classification(name, distance)

            try:
                current_frame_results = self.analysis.results[index]
                current_frame_results.records.append(classification)

            except:
                current_frame_results = FrameResult([classification])
                self.analysis.results.append(current_frame_results)

            index += 1

            if index == 15:
                break

        result_video.release()
        capture.release()
        cv2.destroyAllWindows()
