from pathlib import Path

from gazeclassify.service.model_loader import ModelLoader


def test_return_model_name_from_model_url() -> None:
    classifier = ModelLoader("www.internet.com/model.caffeemodel")
    assert classifier.name == "model.caffeemodel"

def test_return_model_filepath() -> None:

    classifier = ModelLoader(
        "www.internet.com/model.caffeemodel",
        "gazeclassify_data/models"
    )
    expected_filepath = Path.home().joinpath("gazeclassify_data/models/model.caffeemodel")
    assert classifier.file_path == expected_filepath
