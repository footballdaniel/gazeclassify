import logging
import os.path
from dataclasses import dataclass

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.core.services.analysis import Analysis
from gazeclassify.core.services.gaze_distance import DistanceToShape
from gazeclassify.core.services.model_loader import ModelLoader
from gazeclassify.core.services.results import Classification, FrameResult
from gazeclassify.thirdparty.pixellib.helpers import InferSpeed


@dataclass
class SemanticSegmentation:
    analysis: Analysis

    def classify(self, name: str) -> None:

        source_file = str(self.analysis.dataset.world_video.file)

        logger = logging.getLogger(__name__)
        logger.setLevel('INFO')

        model = ModelLoader("https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5",
                            "~/gazeclassify_data/")
        model.download_if_not_available("mask_rcnn_coco.h5")

        segment_image = instance_segmentation(infer_speed=InferSpeed.AVERAGE.value)
        segment_image.load_model(model.file_path)
        target_classes = segment_image.select_target_classes(person=True)

        # SEND FRAME TO WRITER
        video_target = os.path.expanduser("~/gazeclassify_data/") + f"{name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result_video = cv2.VideoWriter(video_target, fourcc, 10,
                                       (self.analysis.dataset.world_video.width,
                                        self.analysis.dataset.world_video.height))

        capture = cv2.VideoCapture(source_file)

        idx = 0
        for record in self.analysis.dataset.records:

            hasframe, frame = capture.read()

            if not hasframe:
                print("Ran out of frames")

            segmask, output = segment_image.segmentFrame(frame, segment_target_classes=target_classes)

            # Only get the masks for all areas of type person
            people_masks = []
            person_class_id = 1
            for index in range(segmask["masks"].shape[-1]):
                if segmask['class_ids'][index] == person_class_id:
                    people_masks.append(segmask["masks"][:, :, index])

            # concatenate all masks (not distinguishing between codes)
            bool_concat = np.any(segmask["masks"], axis=-1)

            binary_image_mask = bool_concat.astype('uint8')
            pixel_distance = DistanceToShape(binary_image_mask)
            pixel_distance.detect_shape(positive_values=1)

            image_width = self.analysis.dataset.world_video.width
            image_height = self.analysis.dataset.world_video.height

            pixel_x = record.gaze.x * image_width
            pixel_y = image_height - (record.gaze.y * image_height)  # flip vertically
            print(f"GAZE: {pixel_x} + {pixel_y}")
            distance = pixel_distance.distance_2d(pixel_x, pixel_y)

            # Make alpha image to rgb
            int_concat = bool_concat.astype('uint8') * 255
            rgb_out = np.dstack([int_concat] * 3)

            # Write video out
            result_video.write(rgb_out)

            # Visualize gaze overlay plot
            img_converted = Image.fromarray(output)
            import matplotlib.pyplot as plt  # type: ignore
            plt.imshow(img_converted)
            plt.scatter(pixel_x, pixel_y)
            # img_converted.show()
            # plt.show()

            idx += 1

            classification = Classification(distance)
            frame_result = FrameResult(index, name, [classification])
            self.analysis.results.append(frame_result)

            if idx == 2:
                print("DEBUG BREAK AFTER 2 FRAMES")
                break

        result_video.release()
        capture.release()
        cv2.destroyAllWindows()
