import numpy as np  # type: ignore

class Test_Measuring2DDistanceGazeTo_Shape:
    def test_read_boolean_2D_mask_and_calculate_distance_gaze_to_closest_pixel(self) -> None:
        image_mask = np.zeros((3, 3))
        image_mask[0, 0] = 1
        image_mask[0, 1] = 1
        gaze = [2, 0]

        detected_shape = np.argwhere(image_mask == 1)
        dist_2 = np.sqrt(np.sum((detected_shape - gaze) ** 2, axis=1))
        distance = np.min(dist_2)
        assert distance == 2

    def test_read_boolean_2D_mask_and_identify_distance_to_gaze_should_return_sqrt2_when_diagnoally(self) -> None:
        mask = np.array(
            [
                [1, 0],
                [0, 0]
            ]
        )
        detected_shape = np.argwhere(mask == 1)
        gaze = [[1, 1]]
        distance = np.linalg.norm(detected_shape - gaze)
        assert distance == np.sqrt(2)
