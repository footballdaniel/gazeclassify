from dataclasses import dataclass
from typing import Optional

from gazeclassify.core.services.video import FrameReader, FrameWriter


@dataclass
class FrameIterator:
    reader: FrameReader
    writer: FrameWriter
    _current_frame: int = 0





