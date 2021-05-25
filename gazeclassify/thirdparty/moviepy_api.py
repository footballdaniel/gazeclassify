import glob
import logging
from dataclasses import dataclass
from pathlib import Path

from moviepy.video.compositing.CompositeVideoClip import clips_array  # type: ignore
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore


@dataclass
class CompositeVideo:
    _video_width: int = 1080
    _video_height: int = 1080

    def index_videos(self, path: Path) -> None:
        self._video_clips = []
        video_files = path.joinpath("*.avi")
        result_videos = glob.glob(str(video_files))
        for result in result_videos:
            current = VideoFileClip(result)
            self._video_width = current.size[0]
            self._video_height = current.size[1]
            self._video_clips.append(current)

    def export(self, filename: Path) -> None:
        if len(self._video_clips) == 0:
            logging.warning("No videos found to export")
            return
        array = clips_array([self._video_clips])
        array.write_videofile(str(filename),  verbose=False, logger=None)
