from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np  # type: ignore


@dataclass
class ImageBuilder:
    width: int = 10
    height: int = 10
    channels: int = 3

    def with_size(self, width: int = 3, height: int = 3) -> ImageBuilder:
        self.width = width
        self.height = height
        return self

    def with_channels(self, channels: int = 3) -> ImageBuilder:
        self.channels = channels
        return self

    def with_default_black_image(self) -> ImageBuilder:
        return ImageBuilder(10, 10, 3)

    def build(self) -> np.ndarray:
        return np.zeros((self.width, self.height, self.channels)).astype('uint8')

@dataclass
class ImageColorBuilder:
    image: np.ndarray

    def __post_init__(self) -> None:
        self.width, self.height, _ = self.image.shape
        self._corner_width = math.ceil(self.width/4)
        self._corner_height = math.ceil(self.height/4)

    def with_top_left_color_black(self) -> ImageColorBuilder:
        self.image[0:self._corner_width, 0:self._corner_height] = 255
        return self

    def build(self) -> np.ndarray:
        return self.image

    def with_bottom_left_color_grey(self) -> ImageColorBuilder:
        self.image[self.width-self._corner_width:, self.height-self._corner_height:] = 125
        return self
