from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupilinvisible import PupilInvisibleLoader
from gazeclassify.service.analysis import Analysis


def test_run_semantic_segmentation_on_two_frame_trial() -> None:
    analysis = Analysis()
    PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data/trial")
    SemanticSegmentation(analysis).classify("Human_Shape")

    import json
    results = json.dumps(analysis.results, default=lambda x: x.__dict__)
    expected_classification = '[{"frame_id": 0, "name": "Human_Shape", "classifications": [{"distance": 64.0788294513828}]}, {"frame_id": 1, "name": "Human_Shape", "classifications": [{"distance": 68.18250922307814}]}]'
    assert results == expected_classification
