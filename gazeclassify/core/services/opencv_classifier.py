from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from gazeclassify.core.services.model_loader import ModelLoader


@dataclass
class OpenCVClassifier:
    model_url: str = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
    proto_file_url: str = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt"

    def _get_keypoints_mapping(self) -> List[str]:
        mapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                   'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        return mapping

    def _get_pose_pairs(self) -> List[List[int]]:
        pose_pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                      [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                      [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                      [2, 17], [5, 16]]
        return pose_pairs

    def _get_mapping_index(self) -> List[List[int]]:
        # Index for the part affinity fields correspoding to the POSE_PAIRS
        # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
        mapping_index = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                  [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                  [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                  [37, 38], [45, 46]]
        return mapping_index

    def _get_colors(self) -> List[List[int]]:
        colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
              [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
              [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
        return colors

    def download_model(self) -> None:
        weights_model = ModelLoader(self.model_url, "~/gazeclassify_data/")
        weights_model.download_if_not_available("pose_iter_440000.caffemodel")

        proto_model = ModelLoader(self.proto_file_url, "~/gazeclassify_data/models")
        proto_model.download_if_not_available("pose_deploy_linevec.prototxt")

        self.weights_file = weights_model.file_path
        self._proto_file = proto_model.file_path

    def classify_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame