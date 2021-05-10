import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from gazeclassify.core.services.analysis import Analysis, PupilInvisibleLoader, SemanticSegmentation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/tests/data")

SemanticSegmentation(analysis).classify("person")

analysis.save_to_json()
