from abc import ABC, abstractmethod

from gazeclassify.service.analysis import Analysis


class Algorithm(ABC):
    _confidence_threshold: float = 0.5

    @property
    def confidence(self) -> float:
        return self._confidence_threshold


    @confidence.setter
    def confidence(self, threshold: float) -> None:
        if threshold < 0:
            self._confidence_threshold = 0
        if threshold > 1:
            self._confidence_threshold = 1

    @property
    @abstractmethod
    def analysis(self) -> Analysis:
        ...

    @abstractmethod
    def classify(self, classifier_name: str) -> None:
        ...
