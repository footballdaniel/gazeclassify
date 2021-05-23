from gazeclassify.domain.serialization import CSVSerializer
from gazeclassify.tests.results_builder import FrameResultBuilder, ClassificationBuilder, InstanceClassificationBuilder


def test_serialize_results_semantic_classification() -> None:
    one_result = (
        FrameResultBuilder()
            .with_frame_id(0)
            .with_name("Semantic")
            .with_classification(ClassificationBuilder().build())
            .build()
    )
    serializer = CSVSerializer()
    dict = serializer._frame_result_to_dict([one_result])
    assert len(dict) == 1


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
    dict = serializer._frame_result_to_dict([first_result, second_result])
    # serializer.encode([first_result, second_result], Path("test.csv"))
    assert len(dict) == 2


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
    dict = serializer._frame_result_to_dict([result])
    # serializer.encode([result], Path("test.csv"))
    assert len(dict) == 2
