from gazeclassify.core.model import Dataset
from pixellib.instance import instance_segmentation  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.pupil_repository import PupilInvisibleRepository


def load_from_pupil_invisible(path: str) -> Dataset:
    PupilInvisibleRepository.load_trial(path)


def segment_with_pixellib() -> None:
    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.process_video(
        "drive/MyDrive/example_pupil.mp4",
        frames_per_second=20,
        output_video_name="output.mp4",
    )
