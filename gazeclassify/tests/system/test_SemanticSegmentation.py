from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupil import PupilLoader
from gazeclassify.service.analysis import Analysis
import json


def test_run_semantic_segmentation_on_two_frame_trial() -> None:
    analysis = Analysis()
    PupilLoader(analysis).from_recordings_folder("gazeclassify/example_data/trial")
    SemanticSegmentation(analysis).classify("Human_Shape")

    results = json.dumps(analysis.results, default=lambda x: x.__dict__)
    expected_classification = '[{"frame_id": 0, "name": "Human_Shape", "classifications": [{"distance": 0.1741947770194003}]}, {"frame_id": 1, "name": "Human_Shape", "classifications": [{"distance": 0.21096293861090812}]}]'
    assert results == expected_classification
