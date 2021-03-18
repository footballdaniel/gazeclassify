from typing import Any

import numpy as np  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.model import Dataset
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository


def load_from_pupil_invisible(path: str) -> Any:
    dataset = PupilInvisibleRepository().load_capture(path)
    return dataset


def segment_with_pixellib() -> None:
    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.process_video(
        "drive/MyDrive/example_pupil.mp4",
        frames_per_second=20,
        output_video_name="output.mp4",
    )
