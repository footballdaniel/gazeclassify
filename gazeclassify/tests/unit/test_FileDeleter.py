from pathlib import Path

from gazeclassify.service.deletion import Deleter


def test_create_file_and_delete() -> None:
    file = Path("gazeclassify/tests/textfile.filetype")
    with open(str(file), 'w') as new_file:
        pass
    Deleter().clear_files(Path('gazeclassify/tests'), "*.filetype")
    assert Path("gazeclassify/tests/textfile.filetype").exists() == False


def test_create_directory_and_delete() -> None:
    directory_path = Path("gazeclassify/example_data/test_directory")
    directory_path.mkdir(parents=True, exist_ok=True)
    Deleter().clear_directory(directory_path)
    assert Path(directory_path).exists() == False
