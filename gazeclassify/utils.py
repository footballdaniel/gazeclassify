import logging
import sys
import time
from contextlib import contextmanager
from logging import Logger
from typing import Iterator, Optional, Set

import numpy as np  # type: ignore


@contextmanager
def performance_logging(description: str, logger: Optional[Logger] = None) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        took = (time.time() - start) * 1000
        unit = "ms"
        if took < 0.1:
            took *= 1000
            unit = "us"
        message = f"{description} took: {took:.2f}{unit}"
        if logger:
            logger.info(message)
        else:
            print(message)


def inspect_recursively(instance: object, recursively_seen: Optional[Set[int]] = set()) -> int:
    if recursively_seen is None:
        recursively_seen = set()
    size = sys.getsizeof(instance)
    obj_id = id(instance)
    if obj_id in recursively_seen:
        return 0
    recursively_seen.add(obj_id)
    if isinstance(instance, np.ndarray):
        size += instance.nbytes
    elif isinstance(instance, dict):
        size += sum([inspect_recursively(v, recursively_seen) for v in instance.values()])
        size += sum([inspect_recursively(k, recursively_seen) for k in instance.keys()])
    elif hasattr(instance, '__dict__'):
        size += inspect_recursively(instance.__dict__, recursively_seen)
    elif hasattr(instance, '__iter__') and not isinstance(instance, (str, bytes, bytearray)):
        size += sum([inspect_recursively(i, recursively_seen) for i in instance])  # type: ignore
    return size


def memory_logging(description: str, instance: object, logger: Optional[Logger] = None) -> None:
    object_size = float(inspect_recursively(instance))
    object_size /= 1000
    unit = "kilobytes"
    message = f"{description} size: {object_size:.1f} {unit}"
    if logger:
        logging.info(message)
    else:
        print(message)
