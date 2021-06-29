from abc import ABC, abstractmethod

from gazeclassify.service.analysis import Analysis


class Algorithm(ABC):
    _minimal_confidence: float = 0.1

    @property
    def minimal_confidence(self) -> float:
        return self._minimal_confidence

    @minimal_confidence.setter
    def minimal_confidence(self, threshold: float) -> None:
        if threshold < 0:
            self._minimal_confidence = 0
        if threshold > 1:
            self._minimal_confidence = 1

    @property
    @abstractmethod
    def analysis(self) -> Analysis:
        ...

    @abstractmethod
    def classify(self, classifier_name: str) -> None:
        ...
