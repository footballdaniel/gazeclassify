import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore
from matplotlib import pyplot as plt, pyplot  # type: ignore
from pixellib.instance import instance_segmentation, custom_segmentation  # type: ignore

from gazeclassify.domain.dataset import DataRecord
from gazeclassify.domain.results import Classification, InstanceClassification
from gazeclassify.service.gaze_distance import DistanceToShape
from gazeclassify.service.model_loader import ModelLoader


class InferSpeed(Enum):
    RAPID = "rapid"
    FAST = "fast"
    AVERAGE = "average"


@dataclass
class Mask:
    name: str
    mask: np.ndarray


@dataclass
class PixellibCustomTensorflowClassifier:
    model_weights: str
    classifier_name: str
    _all_trained_classes: List[str] = field(default_factory=list)
    _boolean_masks: List[Mask] = field(default_factory=list)

    def set_target(self, minimal_confidence: float = 0.7) -> None:
        self.segment_image = custom_segmentation()
        self._all_trained_classes, number_classes = self._get_config()
        self.segment_image.inferConfig(num_classes=number_classes, class_names=self._all_trained_classes)
        self.segment_image.load_model(self.model_weights)

    def classify_frame(self, frame: np.ndarray) -> np.ndarray:
        self._get_frame_size(frame)
        segmentation_mask, _image = self.segment_image.segmentFrame(frame, show_bboxes=True)
        self._boolean_masks = []
        boolean_mask = np.zeros((self.image_height, self.image_width), dtype=bool)

        for index, instance in enumerate(segmentation_mask["class_ids"]):
            class_name = self._all_trained_classes[instance]
            class_mask = segmentation_mask["masks"][:, :, index]
            mask = Mask(class_name, class_mask)
            self._boolean_masks.append(mask)
            boolean_mask = np.logical_or(boolean_mask, class_mask).reshape(self.image_height, self.image_width)

        int_concat = boolean_mask.astype('uint8') * 255
        rgb_out = np.dstack([int_concat] * 3)
        return rgb_out

    def gaze_distance_to_object(self, record: DataRecord) -> List[Classification]:
        classifications = []
        self.pixel_x = record.gaze.x * self.image_width
        self.pixel_y = self.image_height - (record.gaze.y * self.image_height)  # flip vertically
        for index, instance in enumerate(self._boolean_masks):
            binary_image_mask = instance.mask.astype('uint8')
            pixel_distance = DistanceToShape(binary_image_mask)
            pixel_distance.detect_shape(positive_values=1)
            distance = pixel_distance.distance_2d(self.pixel_x, self.pixel_y)
            classification = InstanceClassification(distance, instance.name, index)
            classifications.append(classification)
        return classifications

    def visualize_gaze_overlay(self, image: np.ndarray) -> np.ndarray:
        return ScatterImage(image).scatter(self.pixel_x, self.pixel_y)

    def is_gpu_available(self) -> None:
        list_gpu = tf.config.list_physical_devices('GPU')
        if list_gpu == []:
            logging.info("CUDA not available on GPU, falling back on slower CPU for semantic segmentation")
        else:
            logging.info("Using GPU for instance segmentation")

    def _get_config(self):
        if isinstance(self.classifier_name, list):
            all_trained_classes = self.classifier_name.copy()
            all_trained_classes.insert(0, "BG")
            number_classes = len(self.classifier_name)
        else:
            all_trained_classes = ["BG", self.classifier_name]
            number_classes = 1
        return all_trained_classes, number_classes

    def _create_boolean_mask(self, segmentation_mask: Dict[str, Any]) -> None:
        if len(segmentation_mask["masks"]) == 0:
            segmentation_mask["masks"] = np.zeros((self.image_height, self.image_width, 1), dtype=bool)
        self.boolean_mask = np.any(segmentation_mask["masks"], axis=-1)

    def _get_frame_size(self, frame: np.ndarray) -> None:
        self.image_width = frame.shape[1]
        self.image_height = frame.shape[0]

    def _mask_to_rgb(self) -> np.ndarray:
        int_concat = self.boolean_mask.astype('uint8') * 255
        rgb_out = np.dstack([int_concat] * 3)
        return rgb_out

    def _extract_people_mask(self, segmentation_mask: Dict[str, Any]) -> None:
        people_masks = []
        person_class_id = 1
        for index in range(segmentation_mask["masks"].shape[-1]):
            if segmentation_mask['class_ids'][index] == person_class_id:
                people_masks.append(segmentation_mask["masks"][:, :, index])




@dataclass
class PixellibTensorflowClassifier:
    model_weights: ModelLoader

    def classify_frame(self, frame: np.ndarray) -> np.ndarray:
        self._get_frame_size(frame)
        segmentation_mask, output = self.segment_image.segmentFrame(frame, segment_target_classes=self.target_classes)
        self._create_boolean_mask(segmentation_mask)
        classified_frame = self._mask_to_rgb()
        return classified_frame

    def _create_boolean_mask(self, segmentation_mask: Dict[str, Any]) -> None:
        if len(segmentation_mask["masks"]) == 0:
            segmentation_mask["masks"] = np.zeros((self.image_height, self.image_width, 1), dtype=bool)
        self.boolean_mask = np.any(segmentation_mask["masks"], axis=-1)

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

    def _extract_people_mask(self, segmentation_mask: Dict[str, Any]) -> None:
        people_masks = []
        person_class_id = 1
        for index in range(segmentation_mask["masks"].shape[-1]):
            if segmentation_mask['class_ids'][index] == person_class_id:
                people_masks.append(segmentation_mask["masks"][:, :, index])

    def gaze_distance_to_object(self, record: DataRecord) -> Optional[float]:
        binary_image_mask = self.boolean_mask.astype('uint8')
        pixel_distance = DistanceToShape(binary_image_mask)
        pixel_distance.detect_shape(positive_values=1)
        self.pixel_x = record.gaze.x * self.image_width
        self.pixel_y = self.image_height - (record.gaze.y * self.image_height)  # flip vertically
        distance = pixel_distance.distance_2d(self.pixel_x, self.pixel_y)
        return distance

    def set_target(self, minimal_confidence: float = 0.7) -> None:
        self.segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        self.segment_image.load_model(self.model_weights.file_path, minimal_confidence)
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
