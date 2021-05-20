from pathlib import Path

from gazeclassify.services.model_loader import ClassifierLoader


def test_return_model_name_from_model_url() -> None:
    classifier = ClassifierLoader("www.internet.com/model.caffeemodel")
    assert classifier.name == "model.caffeemodel"

def test_return_model_filepath() -> None:

    classifier = ClassifierLoader(
        "www.internet.com/model.caffeemodel",
        "gazeclassify_data/models"
    )
    expected_filepath = Path.home().joinpath("gazeclassify_data/models/model.caffeemodel")
    assert classifier.file_path == expected_filepath
