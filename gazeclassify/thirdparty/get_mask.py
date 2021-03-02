from gazeclassify.thirdparty.modeldownload import ModelDownload  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore
from gazeclassify.thirdparty.helpers import InferSpeed  # type: ignore
import cv2  # type: ignore

model_file = ModelDownload("mask_rcnn_coco.h5").download()

# Source: https://pixellib.readthedocs.io/en/latest/Image_instance.html

segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
segment_image.load_model(model_file)
target_classes = segment_image.select_target_classes(person=True)
segmask, output = segment_image.segmentImage(
    "image.jpg",
    segment_target_classes=target_classes,
    output_image_name=f"image_new.jpg",
    show_bboxes=True,
)

segmask

cv2.imwrite("image_segmaskoutput.jpg", segmask["masks"].astype(int)[:, :, 1] * 255)
cv2.imwrite("image_maskoutput.jpg", output)

a = 1
