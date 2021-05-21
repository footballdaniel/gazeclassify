from pathlib import Path

from gazeclassify.service.deletion import FileDeleter


def test_create_file_and_delete() -> None:
    file = Path("gazeclassify/tests/textfile.filetype")
    with open(str(file), 'w') as new_file:
        pass

    FileDeleter().clear_files(Path('gazeclassify/tests'), "*.filetype")

    assert Path("gazeclassify/tests/textfile.filetype").exists() == False
