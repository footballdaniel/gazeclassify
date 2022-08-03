import gazeclassify.utils.silent_warnings
import pkg_resources
from typing import Any
from gazeclassify.classifier.instance import InstanceSegmentation
from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupil import PupilLoader
from gazeclassify.service.analysis import Analysis
import gazeclassify.utils.logging_format
from gazeclassify.utils.find import Find

__all__ = [
    'InstanceSegmentation',
    'SemanticSegmentation',
    'PupilLoader',
    'Analysis',
    'example_trial',
]


# Explicit reexport for mypy
def example_trial() -> Any:
    trial_filepath = pkg_resources.resource_filename('gazeclassify', 'example_data/trial')
    return trial_filepath
