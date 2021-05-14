# Disable warnings (Tensorflow for Pixellib)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # type: ignore
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable user warnings (necessary for keras)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn