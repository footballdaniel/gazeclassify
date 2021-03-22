from typing import Union, List, Dict

import numpy as np # type: ignore

# Readables are already higher types. Ideally, Readables would be
# BinaryIO or bytes in the future
Readable = Union[List[float], np.ndarray]
