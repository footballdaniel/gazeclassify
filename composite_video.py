from gazeclassify.thirdparty.moviepy_api import CompositeVideo
import os
from pathlib import Path

video_path = Path(os.path.expanduser("~/gazeclassify_data/videos"))

video = CompositeVideo()
video.index_videos(video_path)
video.export("output_composite.mp4")