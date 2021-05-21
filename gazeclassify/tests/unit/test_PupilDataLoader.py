from gazeclassify.eyetracker.pupilinvisible import PupilDataLoader


class Test_PupilDataLoader_WorldFramesCSV:
    def test_if_can_read_timestamps_from_exemplary_file(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/example_data/")
        assert loader.world_timestamps[0] == 65.300018

    def test_if_reads_correct_number_of_timestamps_lines(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/example_data/")
        assert len(loader.world_timestamps) == 58

    def test_if_last_timestamp_has_correct_value(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/example_data/")
        assert loader.world_timestamps[-1] == 67.200536


class Test_PupilDataLoader_Video:
    def test_if_can_read_number_of_frames_from_exemplary_file(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/example_data/", "frame.mp4")
        assert loader.world_videoframes.shape[0] == 1

    def test_if_video_width_correct(self) -> None:
        loader = PupilDataLoader()
        loader._deserialize_video("gazeclassify/example_data/", "frame.mp4")
        assert loader.world_video_width == 1088

    def test_if_video_height_correct(self) -> None:
        loader = PupilDataLoader()
        loader._deserialize_video("gazeclassify/example_data/", "frame.mp4")
        assert loader.world_video_height == 1080

    def test_read_two_frames_check_dimensions(self) -> None:
        loader = PupilDataLoader()
        loader._deserialize_video("gazeclassify/example_data/", "two_frames.mp4")
        rgb_dimensions = loader.world_videoframes.shape[3]
        width = loader.world_videoframes.shape[1]
        height = loader.world_videoframes.shape[2]
        number_of_frames = loader.world_videoframes.shape[0]
        assert rgb_dimensions == 3
        assert width == 1088
        assert height == 1080
        assert number_of_frames == 2


class Test_PupilDataLoader_GazeData:
    def test_if_can_read_gaze_column_from_textfile(self) -> None:
        loader = PupilDataLoader().load_from_export_folder("gazeclassify/example_data/", "frame.mp4")
        assert len(loader.gaze_x) == 145
