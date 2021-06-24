import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore
from matplotlib import pyplot as plt, pyplot  # type: ignore
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
        self._create_boolean_mask(segmask)
        classified_frame = self._mask_to_rgb()
        return classified_frame

    def _create_boolean_mask(self, segmask: Dict[str, Any]) -> None:
        if len(segmask["masks"]) == 0:
            segmask["masks"] = np.zeros((self.image_height, self.image_width, 1), dtype=bool)
        self.boolean_mask = np.any(segmask["masks"], axis=-1)

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

    def visualize_gaze_overlay(self, image: np.ndarray) -> np.ndarray:
        return ScatterImage(image).scatter(self.pixel_x, self.pixel_y)


@dataclass
class ScatterImage:
    image: np.ndarray

    def scatter(self, x: float, y: float) -> np.ndarray:
        red = (0, 0, 255)  # BGR
        radius = math.ceil(self.image.shape[0] / 100)
        filled = -1
        self.image = cv2.circle(self.image, (int(x), int(y)), radius=radius, color=red, thickness=filled)
        return self.image
