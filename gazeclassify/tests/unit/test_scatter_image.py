from gazeclassify.tests.builder.image_builder import ImageBuilder, ImageColorBuilder
from gazeclassify.thirdparty.pixellib_api import ScatterImage

def test_scatter_gaze_on_top_of_ndarray_return_same_ndarray_size() -> None:
    image = ImageBuilder().with_default_black_image().build()
    colored_image = (
        ImageColorBuilder(image)
            .with_top_left_color_black()
            .with_bottom_left_color_grey()
            .build()
    )
    scattered = ScatterImage(colored_image).scatter(5, 5)
    assert image.shape == scattered.shape
