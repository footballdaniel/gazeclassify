import sys
from contextlib import contextmanager
from time import time


@contextmanager
def performance_logging(description: str, logger=None) -> None:
    start = time.time()
    try:
        yield
    finally:
        took = (time.time() - start) * 1000
        unit = "ms"

        msg = f"{description} took: {took:.2f}{unit}"
        if logger:
            logger.info(msg)
        else:
            print(msg)



def memory_logging(obj, recursively_seen=None) -> None:
    if recursively_seen is None:
        recursively_seen = set()
    size = sys.getsizeof(obj)
    obj_id = id(obj)

    if obj_id in recursively_seen:
        return 0

    recursively_seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([memory_logging(v, recursively_seen) for v in obj.values()])
        size += sum([memory_logging(k, recursively_seen) for k in obj.keys()])

    elif hasattr(obj, '__dict__'):
        size += memory_logging(obj.__dict__, recursively_seen)

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([memory_logging(i, recursively_seen) for i in obj])
    return size


d = {}

print(memory_logging(d))
