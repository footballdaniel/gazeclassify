from pathlib import Path

from gazeclassify.services.analysis import FileDeleter


def test_create_file_and_delete() -> None:
    file = Path("gazeclassify/tests/units/textfile.filetype")
    with open(str(file), 'w') as new_file:
        pass

    FileDeleter().clear_files(Path('gazeclassify/tests/units'), "*.filetype")

    assert Path("gazeclassify/tests/units/textfile.filetype").exists() == False
