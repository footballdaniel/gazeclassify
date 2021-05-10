import logging
import os.path
from dataclasses import dataclass

import cv2  # type: ignore
import numpy as np  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.services.gaze_distance import PixelDistance
from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


@dataclass
class Analysis:
    data_path: str = os.path.expanduser("~/gazeclassify_data/")

    def load_from_pupil_invisible(self, path: str) -> None:
        file_repository = PupilInvisibleRepository(path)
        gaze_data = file_repository.load_gaze_data()
        video_metadata = file_repository.load_video_metadata()
        serializer = PupilDataSerializer()
        self._dataset = serializer.deserialize(gaze_data, video_metadata)


    def classify(self, name: str) -> None:
        source_file = self._dataset.world_video.file

        logger = logging.getLogger(__name__)
        logger.setLevel('INFO')

        segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        model_path = os.path.expanduser("~/gazeclassify_data/mask_rcnn_coco.h5")
        segment_image.load_model(model_path)
        target_classes = segment_image.select_target_classes(person=True)

        # SEND FRAME TO WRITER
        video_target = os.path.expanduser(f"~/gazeclassify_data/{name}.avi")
        result_video = cv2.VideoWriter(video_target, cv2.VideoWriter_fourcc(*'MP4V'), 10,
                                       (self._dataset.world_video.width, self._dataset.world_video.height))

        capture = cv2.VideoCapture(str(source_file))
        for record in self._dataset.records:

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

            binary_image_mask = bool_concat.astype('uint8')
            pixel_distance = PixelDistance(binary_image_mask)
            pixel_distance.detect_shape(positive_values=1)
            distance = pixel_distance.distance_gaze_to_shape(record.gaze.x, record.gaze.y)

            # Make alpha image to rgb
            int_concat = bool_concat.astype('uint8') * 255
            rgb_out = np.dstack([int_concat] * 3)

            # Write video out
            result_video.write(rgb_out)

            break
            # Save result to dataset results

        result_video.release()
        capture.release()
        cv2.destroyAllWindows()
