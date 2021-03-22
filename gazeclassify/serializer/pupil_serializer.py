import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, cast

import numpy as np  # type: ignore

from gazeclassify.core.model import Dataset, Metadata, GazeData, DataRecord
from gazeclassify.core.serialization import Serializer
from gazeclassify.core.utils import Readable

from gazeclassify.utils import memory_logging


class PupilDataSerializer(Serializer):

    def deserialize(self, inputs: Dict[str, Readable]) -> Dataset:
        logger = logging.getLogger(__name__)

        gaze_timestamps_raw = self._readable_to_list_of_floats(inputs['gaze timestamps'])
        gaze_x_raw = self._readable_to_list_of_floats(inputs['gaze x'])
        gaze_y_raw = self._readable_to_list_of_floats(inputs['gaze y'])

        world_video_width = self._readable_to_int(inputs['world video width'])
        world_video_height = self._readable_to_int(inputs['world video height'])
        world_video_framenumber = self._readable_to_int(inputs['world video framenumber'])
        world_video_frames = self._readable_to_ndarray(inputs['world video frames'])
        world_video_timestamps = self._readable_to_list_of_floats(inputs['world timestamps'])

        matcher = TimestampMatcher(world_video_timestamps, gaze_timestamps_raw)
        gaze_x = matcher.match_to_base_framerate(gaze_x_raw)
        gaze_y = matcher.match_to_base_framerate(gaze_y_raw)

        folder_name = self._readable_to_string(inputs['folder name'])

        data_records = []
        for index, _ in enumerate(world_video_timestamps):
            world_timestamp = world_video_timestamps[index]
            video = world_video_frames[index, :, :, :]

            gaze = GazeData(
                gaze_x[index],
                gaze_y[index]
            )

            record = DataRecord(
                world_timestamp,
                video,
                gaze
            )

            data_records.append(record)

        metadata = Metadata(
            folder_name,
            world_video_width,
            world_video_height,
            world_video_framenumber
        )

        dataset = Dataset(data_records, metadata)

        logger = logging.getLogger(__name__)
        logger.setLevel('INFO')
        memory_logging("Size of dataset", dataset, logger)

        return dataset

    def _readable_to_list_of_floats(self, inputs: Readable) -> List[float]:
        transformed_inputs: List[float] = cast(List[float], inputs)
        return transformed_inputs

    def _readable_to_string(self, inputs: Readable) -> str:
        transformed_inputs: str = cast(str, inputs)
        return transformed_inputs

    def _readable_to_ndarray(self, inputs: Readable) -> np.ndarray:
        transformed_inputs: np.ndarray = cast(np.ndarray, inputs)
        return transformed_inputs

    def _readable_to_int(self, inputs: Readable) -> int:
        transformed_inputs: int = cast(int, inputs)
        return transformed_inputs

    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


@dataclass
class TimestampMatcher:
    baseline_timestamps: List[float]
    to_be_matched: List[float]

    def match_to_base_framerate(self, data: List[float]) -> List[float]:
        matched = []
        current_search_index = 0
        for index, current_baseline_timestamp in enumerate(self.baseline_timestamps):

            if current_baseline_timestamp <= self.to_be_matched[current_search_index]:
                matched.append(data[current_search_index])
            else:
                if current_search_index < len(data) - 1:

                    while (current_baseline_timestamp > self.to_be_matched[current_search_index]) & (
                            current_search_index < len(data) - 1):
                        current_search_index += 1

                    matched.append(data[current_search_index])
                else:
                    matched.append(data[-1])

        return matched
