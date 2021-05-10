from pixellib.instance import instance_segmentation, configuration  # type: ignore

from gazeclassify.core.services.analysis import Analysis

analysis = Analysis()

analysis.load_from_pupil_invisible("gazeclassify/tests/data")
analysis.classify("person")
