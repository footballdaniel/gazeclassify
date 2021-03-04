from gazeclassify.core.model import DataRecord, Dataset, GazeData, Metadata, VideoFrame
from gazeclassify.core.data_loader import PupilDataLoader
from pixellib.instance import instance_segmentation  # type: ignore
import numpy as np  # type: ignore


def load_from_pupil_invisible(path: str) -> Dataset:
    data = PupilDataLoader().load_from_export_folder(path)
    print("Has Loaded Data")

    # Require return dataset
    metadata = Metadata("str")
    world_frame = VideoFrame(1, 1, np.array())
    gaze = GazeData(1, 1)
    data_record = DataRecord(0, world_frame, gaze)
    dataset = Dataset([data_record], metadata)
    return dataset


def segment_with_pixellib() -> None:
    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.process_video(
        "drive/MyDrive/example_pupil.mp4",
        frames_per_second=20,
        output_video_name="output.mp4",
    )
