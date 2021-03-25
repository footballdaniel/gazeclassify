from dataclasses import dataclass
from typing import BinaryIO

from gazeclassify.core.model.dataset import Dataset
from gazeclassify.core.services.algorithms import Algorithm
from gazeclassify.core.services.analysis import FrameIterator


@dataclass
class SemanticClassifier:
    _dataset: Dataset
    _frame_iterator: FrameIterator
    _algorithm: Algorithm

    def analyze_frames(self, frame: BinaryIO) -> None:

        for record in self._dataset.records:
            frame = self._frame_iterator.reader.get_frame()




