from gazeclassify.thirdparty.modeldownload import ModelDownload  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore
from enum import Enum
import time
import tensorflow as tf  # type: ignore

tf.get_logger().setLevel("WARNING")

model_file = ModelDownload("mask_rcnn_coco.h5").download()


class InferSpeed(Enum):
    RAPID = "rapid"
    FAST = "fast"
    AVERAGE = "average"


t = time.time()
for i in range(10):
    segment_image = instance_segmentation(infer_speed=InferSpeed.RAPID.value)
    segment_image.load_model(model_file)
    segment_image.segmentImage("image.jpg", output_image_name=f"image_{i}.jpg")
print(time.time() - t)

print("done")
