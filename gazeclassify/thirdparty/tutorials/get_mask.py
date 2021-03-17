from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.thirdparty.helpers import InferSpeed
from gazeclassify.thirdparty.modeldownload import ModelDownload


def main() -> None:
    model_file = ModelDownload("mask_rcnn_coco.h5").download()

    segment_image = instantiate_model(model_file)
    target_classes = segment_image.select_target_classes(person=True)

    # Source: https://pixellib.readthedocs.io/en/latest/Image_instance.html
    segmask, output = extract_mask_and_output(
        segment_image, target_classes, "image.jpg"
    )

    extended_segmask = dilate(segmask)

    export_to_file(segmask, output, extended_segmask)


def dilate(segmask: Dict[str, np.ndarray]) -> np.ndarray:
    black_white_mask = segmask["masks"].astype(int)[:, :, 1] * 255
    black_white_mask = np.asarray(black_white_mask, dtype="uint8")

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel_expansion_x = 5
    kernel_expansion_y = 5
    kernel = np.ones((kernel_expansion_x, kernel_expansion_y), np.uint8)
    dilation = cv2.dilate(black_white_mask, kernel, iterations=1)

    return dilation


def instantiate_model(model_file: Path) -> instance_segmentation:
    segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
    segment_image.load_model(model_file.stem)
    return segment_image


def extract_mask_and_output(
        segment_image: instance_segmentation, targets: Dict[str, str], image_file: str
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    segmask, output = segment_image.segmentImage(
        image_file,
        segment_target_classes=targets,
        show_bboxes=True,
    )
    return segmask, output


def export_to_file(
        segmask: Dict[str, np.ndarray], output: np.ndarray, extended_segmask: np.ndarray
) -> None:
    cv2.imwrite("image_segmaskoutput.jpg", segmask["masks"].astype(int)[:, :, 1] * 255)
    cv2.imwrite("image_coloredoutptu.jpg", output)
    cv2.imwrite("image_extendedsegmask.jpg", extended_segmask)


if __name__ == "__main__":
    main()
