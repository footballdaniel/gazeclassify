import numpy as np  # type: ignore

from gazeclassify.core.services.gaze_distance import PixelDistance


class Test_Measuring2DDistanceGazeTo_Shape:

    def test_read_binary_image_mask_and_calculate_distance_gaze_to_closest_pixel(self) -> None:
        image_mask = np.zeros((3, 3))
        image_mask[0, 0] = 1
        image_mask[0, 1] = 1
        gaze_x = 2
        gaze_y = 0
        pixel_distance = PixelDistance(image_mask)
        pixel_distance.detect_shape(positive_values=1)
        distance = pixel_distance.distance_gaze_to_shape(gaze_x, gaze_y)
        assert distance == 2


    def test_read_boolean_2D_mask_and_identify_distance_to_gaze_should_return_sqrt2_when_diagnoally(self) -> None:
        image_mask = np.array(
            [
                [1, 0],
                [0, 0]
            ]
        )
        gaze_x = 1
        gaze_y = 1
        pixel_distance = PixelDistance(image_mask)
        pixel_distance.detect_shape(positive_values=1)
        distance = pixel_distance.distance_gaze_to_shape(gaze_x, gaze_y)
        assert distance == np.sqrt(2)
