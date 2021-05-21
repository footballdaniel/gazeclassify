from gazeclassify.service.analysis import Analysis
from gazeclassify.eyetracker.pupil_data_loader import PupilInvisibleLoader
from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.classifier.instance import InstanceSegmentation

analysis = Analysis().set_logger("INFO")

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_json()
