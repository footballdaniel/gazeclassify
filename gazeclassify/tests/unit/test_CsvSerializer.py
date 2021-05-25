from gazeclassify.domain.serialization import CSVSerializer
from gazeclassify.tests.builder.results_builder import FrameResultBuilder, ClassificationBuilder, InstanceClassificationBuilder


def test_serialize_results_semantic_classification() -> None:
    one_result = (
        FrameResultBuilder()
            .with_frame_id(0)
            .with_name("Semantic")
            .with_classification(ClassificationBuilder().build())
            .build()
    )
    serializer = CSVSerializer()
    serializer._frame_result_to_dict([one_result])
    assert len(serializer._dict_data) == 1


def test_serialize_results_with_instance_classification_captures_fields() -> None:
    first_result = (
        FrameResultBuilder()
            .with_frame_id(0)
            .with_name("Semantic")
            .with_classification(ClassificationBuilder().build())
            .build()
    )
    second_result = (
        FrameResultBuilder()
            .with_frame_id(0)
            .with_name("Instance")
            .with_classification(InstanceClassificationBuilder().build())
            .build()
    )

    serializer = CSVSerializer()
    serializer._frame_result_to_dict([first_result, second_result])
    # serializer.encode([first_result, second_result], Path("test.csv"))
    assert len(serializer._dict_data) == 2


def test_serialize_results_with_several_classifications_for_instance() -> None:
    result = (
        FrameResultBuilder()
            .with_frame_id(0)
            .with_name("Instance")
            .with_classification(InstanceClassificationBuilder().with_distance(100).with_joint("elbow").build())
            .with_classification(InstanceClassificationBuilder().with_distance(200).with_joint("head").build())
            .build()
    )

    serializer = CSVSerializer()
    serializer._frame_result_to_dict([result])
    # serializer.encode([result], Path("test.csv"))
    assert len(serializer._dict_data) == 2


def test_sort_list_of_dicts_by_dict_key() -> None:
    data = [{"frame": "1", "name": "A"}, {"frame": "0", "name": "B"}]

    serializer = CSVSerializer()
    serializer._dict_data = data

    serializer._sort_dict_by_key("frame")

    assert serializer._dict_data[0]["frame"] == "0"
