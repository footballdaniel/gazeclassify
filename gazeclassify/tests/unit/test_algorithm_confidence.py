from gazeclassify.domain.classification import Algorithm


def override_abstract_methods() -> None:
    Algorithm.__abstractmethods__ = set()  # type: ignore


def test_confidence_has_upper_limit_of_1() -> None:
    override_abstract_methods()
    algorithm = Algorithm()  # type: ignore
    algorithm.minimal_confidence = 2
    assert algorithm.minimal_confidence == 1


def test_confidence_has_lower_limit_of_1() -> None:
    override_abstract_methods()
    algorithm = Algorithm()  # type: ignore
    algorithm.minimal_confidence = -1
    assert algorithm.minimal_confidence == 0
