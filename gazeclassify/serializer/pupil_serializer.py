from dataclasses import dataclass
from typing import Dict, Tuple, List, cast

import numpy as np  # type: ignore

from gazeclassify.core.model import Dataset, Metadata, VideoFrame, GazeData, DataRecord
from gazeclassify.core.serialization import Serializer
from gazeclassify.core.utils import Readable


class PupilDataSerializer(Serializer):

    def deserialize(self, inputs: Dict[str, Readable]) -> Dataset:
        world_timestamps = self._readable_to_list_of_floats(inputs['world timestamps'])
        gaze_timestamps = self._readable_to_list_of_floats(inputs['gaze timestamps'])
        gaze_x = self._readable_to_list_of_floats(inputs['gaze x'])
        gaze_y = self._readable_to_list_of_floats(inputs['gaze y'])

        matcher = TimestampMatcher(world_timestamps, gaze_timestamps)
        gaze_x = matcher.match_to_base_framerate(gaze_x)
        gaze_y = matcher.match_to_base_framerate(gaze_y)

        # placeholder to return dataset
        metadata = Metadata("str")
        world_frame = VideoFrame(1, 1, np.array())
        gaze = GazeData(1, 1)
        data_record = DataRecord(0, world_frame, gaze)
        dataset = Dataset([data_record], metadata)
        return dataset

    def _readable_to_list_of_floats(self, inputs: Readable) -> List[float]:
        transformed_inputs: List[float] = cast(List[float], inputs)
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
