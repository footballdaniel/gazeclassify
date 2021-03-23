from typing import Any

import numpy as np  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.model.dataset import Dataset
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer

def load_from_pupil_invisible(path: str) -> Dataset:
    file_repository = PupilInvisibleRepository(path)
    gaze_data = file_repository.load_gaze_data()
    video_metadata = file_repository.load_video_metadata()

    serializer = PupilDataSerializer()
    dataset = serializer.deserialize(gaze_data, video_metadata)
    return dataset


def segment_humans(dataset: Dataset) -> None:



    for record in Dataset.records:

    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.process_video(
        "drive/MyDrive/example_pupil.mp4",
        frames_per_second=20,
        output_video_name="output.mp4",
    )


load_from_pupil_invisible("gazeclassify/tests/data")