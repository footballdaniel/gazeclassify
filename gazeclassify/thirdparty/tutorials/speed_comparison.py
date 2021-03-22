from pathlib import Path
from gazeclassify.thirdparty.pixellib.modeldownload import ModelDownload
from pixellib.instance import instance_segmentation  # type: ignore
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed
import time
import tensorflow as tf  # type: ignore
import logging

def main() -> None:
    tf.get_logger().setLevel(logging.WARNING)

    model_file = ModelDownload("mask_rcnn_coco.h5").download()

    t = time.time()
    segment_image_ten_times(model_file, "image.jpg")
    print(time.time() - t)

    t = time.time()
    segment_video(model_file, "gazeclassify/tests/data/world.mp4")
    print(time.time() - t)


def segment_image_ten_times(model_file: Path, image_file: str) -> None:
    for i in range(10):
        segment_image = instance_segmentation(infer_speed=InferSpeed.RAPID.value)
        segment_image.load_model(model_file)
        segment_image.segmentImage(image_file, output_image_name=f"image_{i}.jpg")


def segment_video(model_file: Path, video_file: str) -> None:
    segment_video = instance_segmentation()
    segment_video.load_model(model_file)
    segment_video.process_video(
        video_file,
        frames_per_second=20,
        output_video_name="out.mp4",
    )