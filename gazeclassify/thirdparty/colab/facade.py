import os

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation, configuration  # type: ignore

from gazeclassify.core.model.dataset import Dataset, Classification
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer
from gazeclassify.thirdparty.opencv import OpenCVFrameReader, OpenCVFrameWriter
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


def load_from_pupil_invisible(path: str) -> Dataset:
    file_repository = PupilInvisibleRepository(path)
    gaze_data = file_repository.load_gaze_data()
    video_metadata = file_repository.load_video_metadata()

    serializer = PupilDataSerializer()
    dataset = serializer.deserialize(gaze_data, video_metadata)
    return dataset


def classify(dataset: Dataset, name: str) -> None:
    classifcation = Classification(name)
    source_file = dataset.metadata.video_format.file
    frame_reader = OpenCVFrameReader(source_file)

    segment_image = instance_segmentation(infer_speed=InferSpeed.FAST.value)
    model_path = os.path.expanduser("~/gazeclassify_data/mask_rcnn_coco.h5")
    segment_image.load_model(model_path)
    target_classes = segment_image.select_target_classes(person=True)

    frame_reader.open_capture()

    # for record in dataset.records:
    #     frame = frame_reader.get_frame()
    #
    #     # Reconvert to cv2
    #     reconvert = frame.getvalue()
    #     image = cv2.imdecode(np.frombuffer(reconvert, np.uint8), cv2.IMREAD_COLOR)
    #     rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    #     # segment
    #     segmask, output = segment_image.segmentImage(
    #         rgb_image,
    #         segment_target_classes=target_classes,
    #         process_frame=True)
    #     img_converted = Image.fromarray(output)
    #     img_converted.show()
    #
    #     # Only get the masks for person
    #     people_masks = []
    #     person_class_id = 1
    #     for index, mask in enumerate(segmask["masks"]):
    #         if segmask['class_ids'][index] == person_class_id:
    #             people_masks.append(mask)
    #
    #     # concatenate all masks (not distinguishing between codes)
    #     bool_concat = np.any(segmask["masks"], axis=-1)
    #     int_concat = bool_concat.astype(int) * 255
    #
    #     # Extend to 3d rgb image
    #     b = np.repeat(int_concat[:, :, np.newaxis], 3, axis=2)
    #     Image.fromarray(b.astype(np.uint8)).show()
    #
    #     bytesframe = cv2.imencode('.jpg', int_concat)[1].tobytes()
    #
    #     # SEND FRAME TO WRITER
    #
    #     # QUERY GAZE: is on image?
    #
    #     # Save result to dataset results

dataset = load_from_pupil_invisible("gazeclassify/tests/data")
classify(dataset, "person")
