import numpy as np  # type: ignore

from gazeclassify.core.repository import EyeTrackingDataRepository
from gazeclassify.core.utils import Readable
from gazeclassify.serializer.pupil_data_loader import PupilDataLoader


class PupilInvisibleRepository(EyeTrackingDataRepository):
    def load_trial(self, path: str) -> Readable:
        data = PupilDataLoader().load_from_export_folder(path)

        data_dict = {
            "world timestamps": data.world_timestamps,
            "world video height": data.world_video_height,
            "world video width": data.world_video_width,
            "world video framenumber": data.world_video_framenumber,
            "folder name": data.foldername
        }

        return data_dict
