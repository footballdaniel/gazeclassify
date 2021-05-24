import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.domain.dataset import DataRecord
from gazeclassify.service.gaze_distance import DistanceToShape
from gazeclassify.service.model_loader import ModelLoader


class InferSpeed(Enum):
    RAPID = "rapid"
    FAST = "fast"
    AVERAGE = "average"


@dataclass
class PixellibTensorflowClassifier:
    model_weights: ModelLoader

    def classify_frame(self, frame: np.ndarray) -> np.ndarray:
        self._get_frame_size(frame)
        segmask, output = self.segment_image.segmentFrame(frame, segment_target_classes=self.target_classes)
        self._extract_people_mask(segmask)
        self.boolean_mask = np.any(segmask["masks"], axis=-1)
        classified_frame = self._mask_to_rgb()
        return classified_frame

    def is_gpu_available(self) -> None:
        list_gpu = tf.config.list_physical_devices('GPU')
        if list_gpu == []:
            logging.info("CUDA not available on GPU, falling back on slower CPU for semantic segmentation")
        else:
            logging.info("Using GPU for instance segmentation")

    def _get_frame_size(self, frame: np.ndarray) -> None:
        self.image_width = frame.shape[1]
        self.image_height = frame.shape[0]

    def _mask_to_rgb(self) -> np.ndarray:
        int_concat = self.boolean_mask.astype('uint8') * 255
        rgb_out = np.dstack([int_concat] * 3)
        return rgb_out

    def _extract_people_mask(self, segmask: Dict[str, Any]) -> None:
        people_masks = []
        person_class_id = 1
        for index in range(segmask["masks"].shape[-1]):
            if segmask['class_ids'][index] == person_class_id:
                people_masks.append(segmask["masks"][:, :, index])

    def gaze_distance_to_object(self, record: DataRecord) -> Optional[float]:
        binary_image_mask = self.boolean_mask.astype('uint8')
        pixel_distance = DistanceToShape(binary_image_mask)
        pixel_distance.detect_shape(positive_values=1)
        self.pixel_x = record.gaze.x * self.image_width
        self.pixel_y = self.image_height - (record.gaze.y * self.image_height)  # flip vertically
        distance = pixel_distance.distance_2d(self.pixel_x, self.pixel_y)
        return distance

    def set_target(self) -> None:
        self.segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        self.segment_image.load_model(self.model_weights.file_path)
        self.target_classes = self.segment_image.select_target_classes(person=True)

    def _visualize_gaze_overlay(self, output: np.ndarray) -> None:
        img_converted = Image.fromarray(output)
        plt.imshow(img_converted)
        plt.scatter(self.pixel_x, self.pixel_y)
        img_converted.show()
        plt.show()
