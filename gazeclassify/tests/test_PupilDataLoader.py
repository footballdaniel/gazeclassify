from gazeclassify.serializer.pupil_data_loader import PupilDataLoader


class Test_PupilDataLoader_WorldFramesCSV:
    def test_if_can_read_timestamps_from_exemplary_file(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/tests/data/")
        assert loader.world_timestamps[0] == 65.300018

    def test_if_reads_correct_number_of_timestamps_lines(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/tests/data/")
        assert len(loader.world_timestamps) == 58

    def test_if_last_timestamp_has_correct_value(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/tests/data/")
        assert loader.world_timestamps[-1] == 67.200536


class Test_PupilDataLoader_Video:
    def test_if_can_read_number_of_frames_from_exemplary_file(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/tests/data/", "frame.mp4")
        assert loader.world_videoframes.shape[0] == 1

    def test_if_video_width_correct(self) -> None:
        loader = PupilDataLoader()
        loader._deserialize_video("gazeclassify/tests/data/", "frame.mp4")
        assert loader.world_video_width == 1088

    def test_if_video_width_correct(self) -> None:
        loader = PupilDataLoader()
        loader._deserialize_video("gazeclassify/tests/data/", "frame.mp4")
        assert loader.world_video_height == 1080
