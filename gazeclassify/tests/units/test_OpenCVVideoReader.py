import io
from dataclasses import dataclass

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

@dataclass
class VideoHandle:
    stream: cv2.VideoCapture
    width: int
    height: int
    fps: int


class TestOpenCVVideoReader:
    def test_read_two_video_frames_opencv_results_in_two_frames(self) -> None:
        capture = cv2.VideoCapture("gazeclassify/example_data/two_frames.mp4")
        frame_counter = -1
        while True:
            has_frame, frame = capture.read()
            frame_counter += 1
            if not has_frame:
                break
        assert frame_counter == 2

    def test_read_videoframe_to_bytes_and_back_to_image(self) -> None:
        capture = cv2.VideoCapture("gazeclassify/example_data/frame.mp4")

        _, frame = capture.read()

        # https://stackoverflow.com/a/64849668 Convert to bytes
        bytesframe = cv2.imencode('.jpg', frame)[1].tobytes()

        image = cv2.imdecode(np.frombuffer(bytesframe, np.uint8), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_converted = Image.fromarray(rgb_image)

        capture.release()
        assert frame.shape == rgb_image.shape

    def test_read_videoframe_to_bytes_to_bytesIO_back_to_bytes_and_back_to_image(self) -> None:
        capture = cv2.VideoCapture("gazeclassify/example_data/frame.mp4")

        has_frames, frame = capture.read()

        # https://stackoverflow.com/a/64849668 Convert to bytes
        bytesframe = cv2.imencode('.jpg', frame)[1].tobytes()

        # Convert to BytesIO and back
        bytesioframe = io.BytesIO(bytesframe)
        reconvert = bytesioframe.getvalue()

        # Read image with cv2
        image = cv2.imdecode(np.frombuffer(bytesframe, np.uint8), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        capture.release()
        assert frame.shape == rgb_image.shape

    def test_get_video_width_and_height_and_fps(self) -> None:
        capture = cv2.VideoCapture("gazeclassify/example_data/frame.mp4")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_nr = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        assert width == 1088
        assert height == 1080
        assert fps == 1
        assert frame_nr == 1

    def test_write_video(self) -> None:
        capture = cv2.VideoCapture("gazeclassify/example_data/frame.mp4")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        has_frames, frame = capture.read()
        capture.release()

        codec = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter("gazeclassify/example_data/frame_export.mp4", codec, fps, (width, height))
        writer.write(frame)
        writer.release()


