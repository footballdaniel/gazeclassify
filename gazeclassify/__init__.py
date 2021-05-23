# Disable warnings (Tensorflow for Pixellib)
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # type: ignore
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Overwrite logging level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s (%(module)s)')

# Disable user warnings (necessary for keras)
def warn(*args, **kwargs) -> None:  # type: ignore
    pass
import warnings
warnings.warn = warn


