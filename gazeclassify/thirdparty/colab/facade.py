import gazeclassify
from gazeclassify.core.services.analysis import Analysis
from gazeclassify.serializer.pupil_data_loader import PupilInvisibleLoader
from gazeclassify.core.services.semantic_classifier import SemanticSegmentation
from gazeclassify.core.services.instance_classifier import InstanceSegmentation

analysis = Analysis()
PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/tests/data")

SemanticSegmentation(analysis).classify("person")
InstanceSegmentation(analysis).classify("Head")

analysis.save_to_json()
