from gazeclassify.core.models import Dataset, Metadata
from gazeclassify.core.data_loader import PupilDataDeserializer
from pixellib.instance import instance_segmentation  # type: ignore


def load_from_pupil_invisible(path: str) -> Dataset:
    data = PupilDataDeserializer().load_from_export_folder(path)
    print("Has Loaded Data")

    # Require return dataset
    metadata = Metadata("str")
    dataset = Dataset(metadata)
    return dataset


def segment_with_pixellib() -> None:
    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.process_video(
        "drive/MyDrive/example_pupil.mp4",
        frames_per_second=20,
        output_video_name="output.mp4",
    )
