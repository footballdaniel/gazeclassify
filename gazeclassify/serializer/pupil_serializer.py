import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List, BinaryIO

import numpy as np  # type: ignore

from gazeclassify.core.model.dataset import Dataset, Metadata, GazeData, DataRecord, VideoData
from gazeclassify.core.model.serialization import Serializer
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

        logger = logging.getLogger(__name__)
        logger.setLevel('DEBUG')
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

    def _match_data_to_baseline_index(self, current_baseline_timestamp: float, data: List[float]) -> None:
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

    def _has_not_ended(self, data: List[float]) -> bool:
        has_not_ended = self._search_index < len(data) - 1
        return has_not_ended

    def _is_data_timestamp_higher_than(self, current_baseline_timestamp: float) -> bool:
        is_ahead = current_baseline_timestamp <= self.to_be_matched_data[self._search_index]
        return is_ahead
