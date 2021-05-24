import logging
import pkg_resources

from typing import Union
def set_logger(level: Union[int, str]) -> None:
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s (%(module)s)')

set_logger("INFO")

# Explicit reexport for mypy
def example_trial() -> str:
    trial_filepath = pkg_resources.resource_filename('gazeclassify', 'example_data/trial')
    return trial_filepath


# Disable tensorflow warnings when importing tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Only report error messages from tensorflow
import tensorflow as tf  # type: ignore
tf.get_logger().setLevel('ERROR')


# Disable user warnings (necessary for keras)
def warn(*args, **kwargs) -> None:  # type: ignore
    pass
import warnings
warnings.warn = warn

from gazeclassify.classifier.instance import InstanceSegmentation
from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupilinvisible import PupilInvisibleLoader
from gazeclassify.service.analysis import Analysis

__all__ = [
    'InstanceSegmentation',
    'SemanticSegmentation',
    'PupilInvisibleLoader',
    'Analysis',
    'example_trial'
]
