import logging
from typing import Union


def set_logger(level: Union[int, str]) -> None:
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s (%(module)s)')


set_logger("INFO")
