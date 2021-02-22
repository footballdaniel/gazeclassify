from gazepy.core.data_loader import PupilDataLoader


class Test_PupilDataLoader:
    def test_if_can_read_timestamps(self):

        loader = PupilDataLoader().load_from_export_folder("gazepy/tests/data/")

        assert loader.world_timestamps[0] == 2.955558

    def test_if_reads_correct_number_of_timestamps_lines(self):
        pass
