from unittest.mock import MagicMock, PropertyMock


class Test_MockPupilDataDeSerializer:
    def test_instantiate_datarecords(self) -> None:
        # https://docs.python.org/3/library/unittest.mock.html#patch
        # https://docs.python.org/3/library/unittest.mock.html#unittest.mock.PropertyMock

        pupilDataLoader = MagicMock()
        p = PropertyMock(return_value=[0])
        type(pupilDataLoader).world_timestamps = p

        assert pupilDataLoader.world_timestamps == [0]
