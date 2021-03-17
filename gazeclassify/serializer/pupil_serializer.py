from typing import Dict, Tuple

import numpy as np  # type: ignore

from gazeclassify.core.model import Dataset, Metadata, VideoFrame, GazeData, DataRecord
from gazeclassify.core.serialization import Serializer
from gazeclassify.core.utils import Readable


class PupilDataSerializer(Serializer):

    def deserialize(self, inputs: Dict[str, Readable]) -> Dataset:
        metadata = Metadata("str")
        world_frame = VideoFrame(1, 1, np.array())
        gaze = GazeData(1, 1)
        data_record = DataRecord(0, world_frame, gaze)
        dataset = Dataset([data_record], metadata)

        return dataset

    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError
