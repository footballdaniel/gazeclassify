import glob
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileDeleter:

    def clear_files(self, data_path: Path, type: str = "*.mp4") -> None:
        for filepath in glob.glob(os.path.join(str(data_path), type)):
            os.remove(filepath)