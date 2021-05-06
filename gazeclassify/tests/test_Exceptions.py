import pytest


class TestSemanticClassifier:
    def test_raises_exception(self) -> None:
        with pytest.raises(Exception) as exception:
            raise Exception
        assert exception.typename == 'Exception'
