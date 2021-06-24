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