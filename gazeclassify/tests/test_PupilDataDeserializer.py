from gazeclassify.core.data_loader import PupilDataLoader
from gazeclassify.core.serializer import PupilDataSerializer
from unittest.mock import MagicMock, PropertyMock


class Test_PupilDataSerializer:
    def test_instantiate_datarecords(self) -> None:

        # https://docs.python.org/3/library/unittest.mock.html#patch
        # https://docs.python.org/3/library/unittest.mock.html#unittest.mock.PropertyMock

        pupilDataLoader = MagicMock()
        p = PropertyMock(return_value=[0])
        type(pupilDataLoader).world_timestamps = p

        serializer = PupilDataSerializer()
        print("WARNING WRITE TEST")
        assert pupilDataLoader.world_timestamps == [0]

    def test_serialization_video_to_numpy_array(self) -> None:
        pass
