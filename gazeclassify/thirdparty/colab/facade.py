from pixellib.instance import instance_segmentation, configuration  # type: ignore

from gazeclassify.core.services.analysis import Analysis, PupilInvisibleLoader, SemanticSegmentation

analysis = Analysis()
PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/tests/data")
SemanticSegmentation(analysis).classify("person")




