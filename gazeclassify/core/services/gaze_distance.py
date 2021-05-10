from dataclasses import dataclass
from typing import cast, Union, Optional

import numpy as np  # type: ignore


@dataclass
class PixelDistance:
    _boolean_image: np.ndarray

    def detect_shape(self, positive_values: float = 1.) -> None:
        self._shape_area = np.argwhere(self._boolean_image == positive_values)

    def distance_gaze_to_shape(self, gaze_x: float, gaze_y: float) -> Optional[float]:
        if self._shape_area.shape[0] == 0:
            return None

        gaze = [gaze_x, gaze_y]
        euclidean_distance = np.sqrt(np.sum((self._shape_area - gaze) ** 2, axis=1))
        closest_distance = np.min(euclidean_distance)
        distance_pixel = cast(float, closest_distance)
        return distance_pixel
