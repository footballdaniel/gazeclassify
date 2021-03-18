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

        matcher = TimestampMatcher(world_timestamps, gaze_timestamps)
        out = matcher.match_to_baseline(gaze_x)

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

    def match_to_baseline(self, data: List[float]) -> List[float]:
        matched = data

        return matched








