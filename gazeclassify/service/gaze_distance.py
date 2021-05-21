import math
from dataclasses import dataclass
from typing import cast, Optional

import numpy as np  # type: ignore


@dataclass
class DistanceToShape:
    _boolean_image: np.ndarray

    def detect_shape(self, positive_values: float = 1.) -> None:
        self._shape_area = np.argwhere(self._boolean_image == positive_values)

    def distance_2d(self, gaze_x: float, gaze_y: float) -> Optional[float]:
        if self._shape_area.shape[0] == 0:
            return None

        gaze = [gaze_x, gaze_y]
        euclidean_distance = np.sqrt(np.sum((self._shape_area - gaze) ** 2, axis=1))
        closest_distance = np.min(euclidean_distance)
        distance_pixel = cast(float, closest_distance)
        return distance_pixel


@dataclass
class DistanceToPoint:
    point_x: float
    point_y: float

    def distance_2d(self, gaze_x: float, gaze_y: float) -> Optional[float]:
        if (self.point_x == None) | (self.point_y == None):
            return None

        point = [self.point_x, self.point_y]
        gaze = [gaze_x, gaze_y]
        euclidean_distance = math.sqrt(((gaze[0] - point[0]) ** 2) + ((gaze[1] - point[1]) ** 2))
        return euclidean_distance
