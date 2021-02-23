from gazepy.core.data_loader import PupilDataDeserializer


class Test_PupilDataDeserializer_WorldFramesCSV:
    def test_if_can_read_timestamps_from_exemplary_file(self):

        loader = PupilDataDeserializer().load_from_export_folder("gazepy/tests/data/")

        assert loader.world_timestamps[0] == 65.300018

    def test_if_reads_correct_number_of_timestamps_lines(self):
        loader = PupilDataDeserializer().load_from_export_folder("gazepy/tests/data/")

        assert len(loader.world_timestamps) == 58

    def test_if_last_timestamp_has_correct_value(self):

        loader = PupilDataDeserializer().load_from_export_folder("gazepy/tests/data/")

        assert loader.world_timestamps[-1] == 67.200536


class Test_PupilDataDeserializer_Video:
    def test_if_can_read_first_frame_to_array_from_exemplary_file(self):
        pass
