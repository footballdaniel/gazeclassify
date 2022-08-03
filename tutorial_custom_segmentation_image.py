import cv2
from matplotlib import image

from gazeclassify.domain.dataset import DataRecord, GazeData
from gazeclassify.thirdparty.pixellib_api import PixellibCustomTensorflowClassifier

# Learnings: the infer config needs to take exactly the right number of classes in
# (all that it was trained on. Each entry is four bytes)

import pixellib


def pixellib_image_segmentation():
    from pixellib.instance import custom_segmentation
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=5, class_names=["BG", "Mat", "Vault", "Trampoline", "Queue", "Jumper"])
    segment_image.load_model("mask_rcnn_model.041-0.951009.h5")
    segment_image.segmentImage("image.png", show_bboxes=True, output_image_name="sample_out.jpg")


def gazeclassify_image_segmentation():
    global frame
    frame = cv2.imread("image.png")
    classifier = PixellibCustomTensorflowClassifier("mask_rcnn_model.041-0.951009.h5",
                                                    ["Mat", "Vault", "Trampoline", "Queue", "Jumper"])
    classifier.set_target()
    classifier.classify_frame(frame)
    record = DataRecord(None, None, GazeData(0.5, 0.5))
    result = classifier.gaze_distance_to_object(record)
    print(result[0])

pixellib_image_segmentation()
gazeclassify_image_segmentation()
