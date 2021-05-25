import logging
from pathlib import Path
from typing import Any

from gazeclassify.thirdparty.moviepy_api import CompositeVideo


def test_make_composite_video_from_videos_in_folder() -> None:
    path = Path("gazeclassify/example_data/video_output")
    video = CompositeVideo()
    video.index_videos(path)
    assert len(video._video_clips) == 2


LOGGER = logging.getLogger(__name__)


def test_if_no_videos_in_results_raise_warning(caplog: Any) -> None:
    with caplog.at_level(logging.WARNING):
        path = Path("Path_with_no_videos")
        video = CompositeVideo()
        video.index_videos(path)
        video.export(Path.home())
    assert 'No videos found to export' in caplog.text
