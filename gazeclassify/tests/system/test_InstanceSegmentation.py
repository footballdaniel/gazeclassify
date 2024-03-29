from gazeclassify.classifier.instance import InstanceSegmentation
from gazeclassify.eyetracker.pupil import PupilLoader
from gazeclassify.service.analysis import Analysis
import json


def test_run_instance_segmentation_on_two_frame_trial() -> None:
    analysis = Analysis()
    PupilLoader(analysis).from_recordings_folder("gazeclassify/example_data/trial")
    InstanceSegmentation(analysis).classify("Human_Joints")

    results = json.dumps(analysis.results, default=lambda x: x.__dict__)

    expected_string = '[{"frame_id": 0, "name": "Human_Joints", "classifications": [{"distance": 79.1827334889696, "name": "Neck", "id": 0}, {"distance": 309.0170428717914, "name": "Right Shoulder", "id": 0}, {"distance": 454.36084688730887, "name": "Right Elbow", "id": 0}, {"distance": 569.8247661952788, "name": "Right Wrist", "id": 0}, {"distance": 158.31715337975731, "name": "Left Shoulder", "id": 0}, {"distance": 433.9555175875743, "name": "Left Elbow", "id": 0}, {"distance": 296.5253160707722, "name": "Left Wrist", "id": 0}, {"distance": 339.8611533037756, "name": "Right Eye", "id": 0}, {"distance": 305.00686619300035, "name": "Left Eye", "id": 0}, {"distance": 342.5454657815682, "name": "Right Ear", "id": 0}, {"distance": 251.4997232367386, "name": "Left Ear", "id": 0}]}, {"frame_id": 1, "name": "Human_Joints", "classifications": [{"distance": 61.21012691701699, "name": "Neck", "id": 0}, {"distance": 236.3943736142106, "name": "Right Shoulder", "id": 0}, {"distance": 191.30898535977087, "name": "Left Shoulder", "id": 0}, {"distance": 284.0567431837079, "name": "Right Eye", "id": 0}, {"distance": 251.0558481496244, "name": "Left Eye", "id": 0}, {"distance": 289.9618470634228, "name": "Right Ear", "id": 0}, {"distance": 222.61013957519953, "name": "Left Ear", "id": 0}]}]'
    assert results == expected_string
