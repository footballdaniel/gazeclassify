from dataclasses import dataclass

from gazeclassify.core.services.video import FrameReader, FrameWriter


@dataclass
class AnalysisState:
    reader: FrameReader
    writer: FrameWriter
