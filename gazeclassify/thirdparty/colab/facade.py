import os

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation, configuration  # type: ignore

from gazeclassify.core.model.dataset import Dataset
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


def load_from_pupil_invisible(path: str) -> Dataset:
    file_repository = PupilInvisibleRepository(path)
    gaze_data = file_repository.load_gaze_data()
    video_metadata = file_repository.load_video_metadata()

    serializer = PupilDataSerializer()
    dataset = serializer.deserialize(gaze_data, video_metadata)
    return dataset


def classify(dataset: Dataset, name: str) -> None:
    source_file = dataset.world_video.file

    segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
    model_path = os.path.expanduser("~/gazeclassify_data/mask_rcnn_coco.h5")
    segment_image.load_model(model_path)
    target_classes = segment_image.select_target_classes(person=True)

    # SEND FRAME TO WRITER
    video_target = os.path.expanduser(f"~/gazeclassify_data/{name}.avi")
    result_video = cv2.VideoWriter(video_target, cv2.VideoWriter_fourcc(*'MP4V'), 10,
                                   (dataset.world_video.width, dataset.world_video.height))

    capture = cv2.VideoCapture(str(source_file))
    for record in dataset.records:

        hasframe, frame = capture.read()

        if not hasframe:
            print("Ran out of frames")

        segmask, output = segment_image.segmentFrame(frame, segment_target_classes=target_classes)

        # img_converted = Image.fromarray(output)
        # img_converted.show()

        # Only get the masks for all areas of type person
        people_masks = []
        person_class_id = 1
        for index in range(segmask["masks"].shape[-1]):
            if segmask['class_ids'][index] == person_class_id:
                people_masks.append(segmask["masks"][:, :, index])

        # concatenate all masks (not distinguishing between codes)
        bool_concat = np.any(segmask["masks"], axis=-1)
        int_concat = bool_concat.astype(int) * 255

        # Get distance to mask
        detected_shape = np.argwhere(int_concat == 255)
        # Get gaze and calculate distance from mask
        gaze = [record.gaze.x, record.gaze.y]
        dist_2 = np.sqrt(np.sum((detected_shape - gaze) ** 2, axis=1))
        distance = np.min(dist_2)

        # Make alpha image to rgb
        rgb_out = np.dstack([int_concat]*3)
        rgb_uint_out = rgb_out.astype('uint8')

        # Write video out
        result_video.write(rgb_uint_out)


        break
        # Save result to dataset results

    result_video.release()
    capture.release()
    cv2.destroyAllWindows()


dataset = load_from_pupil_invisible("gazeclassify/tests/data")
classify(dataset, "person")
