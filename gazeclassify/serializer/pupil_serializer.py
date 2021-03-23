import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, BinaryIO

import numpy as np  # type: ignore

from gazeclassify.core.model import Dataset, Metadata, GazeData, DataRecord, VideoFrame
from gazeclassify.core.serialization import Serializer
from gazeclassify.core.video import FrameSeeker
from gazeclassify.utils import memory_logging


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
        textio = io.TextIOWrapper(self.file_stream, encoding='utf-8')
        csv_reader = csv.reader(textio, delimiter=",")
        self.line_count = 0
        for row in csv_reader:
            self._read_gaze_data_lines(self.line_count, row)

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


class PupilDataSerializer(Serializer):

    def deserialize(
            self,
            inputs: Dict[str, BinaryIO],
            video_metadata: Dict[str, int],
            video_capture: FrameSeeker) -> Dataset:
        timestamps = TimestampsDeserializer(inputs['world timestamps'])
        timestamps.deserialize()
        world_video_timestamps = timestamps.world_timestamps

        gaze = GazeDeserializer(inputs['gaze location'])
        gaze.deserialize()
        gaze_timestamps_raw = gaze.gaze_timestamps_raw
        gaze_x_raw = gaze.gaze_x_raw
        gaze_y_raw = gaze.gaze_y_raw

        matcher = TimestampMatcher(world_video_timestamps, gaze_timestamps_raw)
        gaze_x = matcher.match_to_base_frame_rate(gaze_x_raw)
        gaze_y = matcher.match_to_base_frame_rate(gaze_y_raw)

        world_video_width = video_metadata['width']
        world_video_height = video_metadata['height']
        world_video_frame_rate = video_metadata['frame rate']
        world_video_frame_number = video_metadata['frame number']

        data_records = []
        for index, _ in enumerate(world_video_timestamps):
            world_timestamp = world_video_timestamps[index]

            video_frame = VideoFrame(index)

            gaze_data = GazeData(
                gaze_x[index],
                gaze_y[index]
            )

            record = DataRecord(
                world_timestamp,
                video_frame,
                gaze_data
            )

            data_records.append(record)

        metadata = Metadata(
            "TOBEDONE FOLDERNAME",
            world_video_width,
            world_video_height,
            world_video_frame_number,
            world_video_frame_rate
        )

        video = video_capture

        dataset = Dataset(data_records, metadata, video)

        logger = logging.getLogger(__name__)
        logger.setLevel('INFO')
        memory_logging("Size of deserialized dataset", dataset, logger)

        return dataset

    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


@dataclass
class TimestampMatcher:
    baseline_timestamps: List[float]
    to_be_matched_data: List[float]
    matched_timestamps: List[float] = field(default_factory=list)
    _search_index: int = 0

    def match_to_base_frame_rate(self, data: List[float]) -> List[float]:
        for current_baseline_timestamp in self.baseline_timestamps:
            self._match_data_to_baseline_index(current_baseline_timestamp, data)
        return self.matched_timestamps

    def _match_data_to_baseline_index(self, current_baseline_timestamp, data):
        if self._is_data_timestamp_higher_than(current_baseline_timestamp):
            self.matched_timestamps.append(data[self._search_index])
        else:
            if self._has_not_ended(data):
                while (current_baseline_timestamp > self.to_be_matched_data[self._search_index]) & (
                        self._search_index < len(data) - 1):
                    self._search_index += 1

                self.matched_timestamps.append(data[self._search_index])
            else:
                self.matched_timestamps.append(data[-1])

    def _has_not_ended(self, data: float) -> bool:
        has_not_ended = self._search_index < len(data) - 1
        return has_not_ended

    def _is_data_timestamp_higher_than(self, current_baseline_timestamp: float) -> bool:
        is_ahead = current_baseline_timestamp <= self.to_be_matched_data[self._search_index]
        return is_ahead
