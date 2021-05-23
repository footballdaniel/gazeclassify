from gazeclassify.classifier.instance import InstanceSegmentation
from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupilinvisible import PupilInvisibleLoader
from gazeclassify.service.analysis import Analysis

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data/trial")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
