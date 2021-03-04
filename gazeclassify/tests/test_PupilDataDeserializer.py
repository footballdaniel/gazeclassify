from _typeshed import FileDescriptorLike
from gazeclassify.core.data_loader import PupilDataLoader
from gazeclassify.core.serializer import PupilDataSerializer


class Test_PupilDataSerializer:
    def test_instantiate_datarecords(self) -> None:

        serializer = PupilDataSerializer()
        assert 1 == 1

    def test_serialization_video_to_numpy_array(self) -> None:
        pass
