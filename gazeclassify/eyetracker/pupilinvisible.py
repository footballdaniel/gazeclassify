from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, BinaryIO, Dict, Tuple

import cv2  # type: ignore
import ffmpeg  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.domain.dataset import Dataset, GazeData, DataRecord, VideoData, Metadata
from gazeclassify.domain.repository import EyeTrackingDataRepository
from gazeclassify.domain.serialization import Serializer
from gazeclassify.service.analysis import Analysis
from gazeclassify.service.timestamp_matcher import TimestampMatcher


@dataclass
class PupilDataLoader:
    _foldername: str = "Video_000"
    _world_video_height: int = 0
    _world_video_width: int = 0
    _world_video_frame_number: int = 0
    _world_timestamps: List[float] = field(default_factory=list)
    _gaze_x: List[float] = field(default_factory=list)
    _gaze_y: List[float] = field(default_factory=list)
    _gaze_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        return self._world_timestamps

    @property
    def world_videoframes(self) -> np.ndarray:
        return self._world_videoframes

    @property
    def world_video_height(self) -> int:
        return self._world_video_height

    @property
    def world_video_width(self) -> int:
        return self._world_video_width

    @property
    def world_video_framenumber(self) -> int:
        return self._world_video_frame_number

    @property
    def foldername(self) -> str:
        return self._foldername

    @property
    def gaze_x(self) -> List[float]:
        return self._gaze_x

    @property
    def gaze_y(self) -> List[float]:
        return self._gaze_y

    @property
    def gaze_timestamps(self) -> List[float]:
        return self._gaze_timestamps


    def access_gaze_file(self, path: str) -> BinaryIO:
        full_filename = self._get_gaze_positions_filepath(path)
        gaze_stream = open(full_filename, 'rb')
        return gaze_stream

    def access_world_timestamps(self, path: str) -> BinaryIO:
        full_filename = self._get_world_timestamps_filepath(path)
        timestamps_stream = open(full_filename, 'rb')
        return timestamps_stream

    def load_from_export_folder(self, path: str, default_video_name: str = "world.mp4") -> PupilDataLoader:
        timestamps_file = self._get_world_timestamps_filepath(path)
        self._get_folder_name(path)
        self._deserialize_world_timestamps(timestamps_file)
        self._deserialize_video(path, default_video_name)
        self._deserialize_gaze_data(path)
        return self

    def _get_folder_name(self, path: str) -> None:
        path_to_folder = Path(path)
        self._foldername = path_to_folder.stem

    def _get_world_timestamps_filepath(self, path: str) -> Path:
        folder = Path(path)
        full_filename = Path.joinpath(folder, "world_timestamps.csv")
        return full_filename

    def _deserialize_world_timestamps(self, timestamps_file: Path) -> None:
        if timestamps_file.exists():
            with open(timestamps_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                self.line_count = 0
                for row in csv_reader:
                    self._read_world_timestamps_lines(self.line_count, row)
        else:
            raise FileNotFoundError(
                "Could not find the file world_timestamps.csv in folder"
            )

    def _read_world_timestamps_lines(self, line_count: int, row: List[str]) -> None:
        if self.line_count == 0:
            self.line_count += 1
        else:
            self._world_timestamps.append(float(row[0]))
            self.line_count += 1

    def _deserialize_video(self, path: str, default_video_name: str) -> None:
        folder = Path(path)
        full_filename = Path.joinpath(folder, default_video_name)
        self._ffmpeg_decode_size(full_filename)
        framebuffer = self._ffmpeg_decode(full_filename)
        frames = np.frombuffer(framebuffer, np.uint8).reshape([-1, self._world_video_width, self.world_video_height, 3])
        self._world_videoframes = frames

    def _ffmpeg_decode_size(self, full_filename: Path) -> None:
        probe = ffmpeg.probe(full_filename)
        video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video['width'])  # type: ignore
        height = int(video['height'])  # type: ignore

        self._world_video_height = height
        self._world_video_width = width

    def _ffmpeg_decode(self, full_filename: Path) -> Any:
        replay_rate_input = 1
        log_level = 0
        frame_buffer, _ = (
            ffmpeg.input(full_filename, r=replay_rate_input)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", **{"loglevel": log_level})
                .run(capture_stdout=True)
        )
        return frame_buffer

    def _deserialize_gaze_data(self, path: str) -> None:
        full_filename = self._get_gaze_positions_filepath(path)
        if full_filename.exists():
            with open(full_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                self.line_count = 0
                for row in csv_reader:
                    self._read_gaze_data_lines(self.line_count, row)
        else:
            raise FileNotFoundError(
                "Could not find the file world_timestamps.csv in folder"
            )

    def _read_gaze_data_lines(self, line_count: int, row: List[str]) -> None:
        if self.line_count == 0:
            self._column_gaze_x = [i for i, element in enumerate(row) if 'norm_pos_x' in element][0]
            self._column_gaze_y = [i for i, element in enumerate(row) if 'norm_pos_y' in element][0]
            self.line_count += 1
        else:
            self._gaze_x.append(float(row[self._column_gaze_x]))
            self._gaze_y.append(float(row[self._column_gaze_y]))
            self._gaze_timestamps.append(float(row[0]))
            self.line_count += 1

    def _get_gaze_positions_filepath(self, path: str) -> Path:
        folder = Path(path)
        full_filename = Path.joinpath(folder, "gaze_positions.csv")
        return full_filename


@dataclass
class PupilInvisibleLoader:
    analysis: Analysis

    def from_trial_folder(self, path: str) -> None:
        file_repository = PupilInvisibleRepository(path)
        gaze_data = file_repository.load_gaze_data()
        video_metadata = file_repository.load_video_metadata()
        serializer = PupilInvisibleSerializer()
        self.analysis.dataset = serializer.deserialize(gaze_data, video_metadata)


@dataclass
class PupilInvisibleRepository(EyeTrackingDataRepository):
    folder_path: str

    def load_gaze_data(self) -> Dict[str, BinaryIO]:
        loader = PupilDataLoader()

        gaze_filestream = loader.access_gaze_file(self.folder_path)
        world_timestamps_filestream = loader.access_world_timestamps(self.folder_path)

        filestream_dict = {
            "gaze location": gaze_filestream,
            "world timestamps": world_timestamps_filestream
        }
        return filestream_dict

    def load_video_metadata(self) -> Dict[str, str]:
        video_file = self._get_file_name(self.folder_path)
        capture = cv2.VideoCapture(video_file)
        width = str(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = str(capture.get(cv2.CAP_PROP_FPS))
        frame_number = str(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        folder_path = self.folder_path
        video_metadata = {
            'width': width,
            'height': height,
            'frame rate': frame_rate,
            'frame number': frame_number,
            'world video file': video_file,
            'folder path': folder_path
        }
        return video_metadata

    def _get_file_name(self, path: str) -> str:
        filename = path + "/world.mp4"
        return filename


@dataclass
class TimestampsDeserializer:
    file_stream: BinaryIO
    _world_timestamps: List[float] = field(default_factory=list)

    @property
    def world_timestamps(self) -> List[float]:
        timestamps: List[float] = self._world_timestamps
        return timestamps

    def deserialize(self) -> None:
        textio = io.TextIOWrapper(self.file_stream, encoding='utf-8')
        csv_reader = csv.reader(textio, delimiter=",")
        self.line_count = 0
        for row in csv_reader:
            self._read_world_timestamps_lines(self.line_count, row)

    def _read_world_timestamps_lines(self, line_count: int, row: List[str]) -> None:
        if self.line_count == 0:
            self.line_count += 1
        else:
            self._world_timestamps.append(float(row[0]))
            self.line_count += 1


@dataclass
class GazeDeserializer:
    file_stream: BinaryIO
    _gaze_timestamps: List[float] = field(default_factory=list)
    _gaze_x: List[float] = field(default_factory=list)
    _gaze_y: List[float] = field(default_factory=list)
    _line_count: int = 0

    @property
    def gaze_timestamps_raw(self) -> List[float]:
        return self._gaze_timestamps

    @property
    def gaze_x_raw(self) -> List[float]:
        return self._gaze_x

    @property
    def gaze_y_raw(self) -> List[float]:
        return self._gaze_y

    def deserialize(self) -> None:
        text = io.TextIOWrapper(self.file_stream, encoding='utf-8')
        csv_reader = csv.reader(text, delimiter=",")
        for row in csv_reader:
            self._read_gaze_data_lines(self._line_count, row)

    def _read_gaze_data_lines(self, line_count: int, row: List[str]) -> None:
        if self._line_count == 0:
            self._column_gaze_x = [i for i, element in enumerate(row) if 'norm_pos_x' in element][0]
            self._column_gaze_y = [i for i, element in enumerate(row) if 'norm_pos_y' in element][0]
            self._line_count += 1
        else:
            self._gaze_x.append(float(row[self._column_gaze_x]))
            self._gaze_y.append(float(row[self._column_gaze_y]))
            self._gaze_timestamps.append(float(row[0]))
            self._line_count += 1


class PupilInvisibleSerializer(Serializer):

    def deserialize(self, gaze_data: Dict[str, BinaryIO], video_metadata: Dict[str, str]) -> Dataset:
        timestamps = TimestampsDeserializer(gaze_data['world timestamps'])
        timestamps.deserialize()
        world_video_timestamps = timestamps.world_timestamps

        gaze = GazeDeserializer(gaze_data['gaze location'])
        gaze.deserialize()
        gaze_timestamps_raw = gaze.gaze_timestamps_raw
        gaze_x_raw = gaze.gaze_x_raw
        gaze_y_raw = gaze.gaze_y_raw

        matcher = TimestampMatcher(world_video_timestamps, gaze_timestamps_raw)
        gaze_x = matcher.match_to_base_frame_rate(gaze_x_raw)
        matcher = TimestampMatcher(world_video_timestamps, gaze_timestamps_raw)
        gaze_y = matcher.match_to_base_frame_rate(gaze_y_raw)

        world_video_width = int(float(video_metadata['width']))
        world_video_height = int(float(video_metadata['height']))
        world_video_frame_rate = int(float(video_metadata['frame rate']))
        world_video_frame_number = int(float(video_metadata['frame number']))
        world_video_file = video_metadata['world video file']

        recording_name = video_metadata['folder path']
        video_path = Path(world_video_file)

        data_records = []
        for index, _ in enumerate(world_video_timestamps):
            world_timestamp = world_video_timestamps[index]

            gaze_location = GazeData(
                gaze_x[index],
                gaze_y[index]
            )

            record = DataRecord(
                world_timestamp,
                index,
                gaze_location
            )

            data_records.append(record)

        world_video = VideoData(
            file=video_path,
            width=world_video_width,
            height=world_video_height,
            frame_number=world_video_frame_number,
            frame_rate=world_video_frame_rate
        )

        metadata = Metadata(
            recording_name=recording_name,
        )

        dataset = Dataset(data_records, metadata, world_video)
        return dataset

    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError