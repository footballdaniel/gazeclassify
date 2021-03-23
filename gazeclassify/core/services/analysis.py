from dataclasses import dataclass
from typing import Optional

from gazeclassify.core.services.video import FrameReader, FrameWriter


@dataclass
class AnalysisState:
    reader: FrameReader
    writer: Optional[FrameWriter]
