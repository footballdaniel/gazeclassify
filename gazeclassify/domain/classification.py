from abc import ABC, abstractmethod

from gazeclassify.service.analysis import Analysis


class Algorithm(ABC):

    @property
    @abstractmethod
    def analysis(self) -> Analysis:
        ...

    @abstractmethod
    def classify(self, classifier_name: str) -> None:
        ...