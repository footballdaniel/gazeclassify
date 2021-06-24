import time
from contextlib import contextmanager
from logging import Logger
from typing import Iterator, Optional

from tqdm import tqdm  # type: ignore


class ProgressBar(tqdm):  # type: ignore
    """
    Source: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


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
