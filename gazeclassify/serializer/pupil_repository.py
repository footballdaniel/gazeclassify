import numpy as np # type: ignore

from gazeclassify.serializer.pupil_data_loader import PupilDataLoader
from gazeclassify.core.model import Dataset, VideoFrame, Metadata, GazeData, DataRecord
from gazeclassify.core.repository import EyeTrackingDataRepository


class PupilInvisibleRepository(EyeTrackingDataRepository):
    def load_trial(self, path: str) -> Dataset:
        data = PupilDataLoader().load_from_export_folder(path)


        # TODO: Manual assembly of Dataset
        metadata = Metadata("str")
        world_frame = VideoFrame(1, 1, np.array())
        gaze = GazeData(1, 1)
        data_record = DataRecord(0, world_frame, gaze)
        dataset = Dataset([data_record], metadata)

        return dataset
