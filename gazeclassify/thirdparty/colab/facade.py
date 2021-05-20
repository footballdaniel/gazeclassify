from gazeclassify.services.analysis import Analysis
from gazeclassify.eyetracker.pupil_data_loader import PupilInvisibleLoader
from gazeclassify.services.semantic_classifier import SemanticSegmentation
from gazeclassify.services.instance_classifier import InstanceSegmentation

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_json()
