__version__ = "0.5"

from .utils import set_logger
set_logger("INFO")

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
from gazeclassify.utils import example_trial

# Explicit reexport for mypy
__all__ = [
    'InstanceSegmentation',
    'SemanticSegmentation',
    'PupilInvisibleLoader',
    'Analysis',
    'example_trial'
]

