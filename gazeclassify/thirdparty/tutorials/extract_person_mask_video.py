import logging

import tensorflow as tf  # type: ignore
from pixellib.instance import instance_segmentation, configuration  # type: ignore
from pixellib.mask_rcnn import MaskRCNN  # type: ignore

from gazeclassify.thirdparty.pixellib.modeldownload import ModelDownload

def main() -> None:
    tf.get_logger().setLevel(logging.WARNING)
    model_file = ModelDownload("mask_rcnn_coco.h5").download()
    segment_video = instance_segmentation()
    segment_video.load_model(model_file)

    coco_config = configuration(BACKBONE="resnet101", NUM_CLASSES=81, class_names=["BG"], IMAGES_PER_GPU=1,
                                DETECTION_MIN_CONFIDENCE=0.10, IMAGE_MAX_DIM=1024, IMAGE_MIN_DIM=800,
                                IMAGE_RESIZE_MODE="square", GPU_COUNT=1)

    # Hack the rcnn to accept my config. it overrides part of the load_model() function
    segment_video.model = MaskRCNN(mode="inference", model_dir=segment_video.model_dir, config=coco_config)

    segment_video.process_video("gazeclassify/tests/example_data/world.mp4", frames_per_second=20, output_video_name="out.mp4")

    a = 1


if __name__ == "__main__":
    main()
