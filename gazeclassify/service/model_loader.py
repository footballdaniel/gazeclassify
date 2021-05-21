from dataclasses import dataclass
from pathlib import Path
from urllib import request

from tqdm import tqdm  # type: ignore

from gazeclassify.utils import ProgressBar


@dataclass
class ModelLoader:
    model_url: str = ""
    download_path: str = "gazeclassify_data/models"

    @property
    def file_path(self) -> Path:
        return (
            Path
                .home()
                .joinpath(self.download_path)
                .joinpath(self.name)
        )

    @property
    def name(self) -> str:
        return Path(self.model_url).name

    def download_if_not_available(self) -> None:
        self._try_make_folder()

        if not self.file_path.exists():
            self._download_classifier()

    def _try_make_folder(self) -> None:
        Path.mkdir(
            Path.home().joinpath(self.download_path),
            parents=True,
            exist_ok=True
        )

    def _download_classifier(self) -> None:
        with ProgressBar(
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=f"Downloading Model: {self.name}"
        ) as t:
            request.urlretrieve(
                self.model_url,
                self.file_path,
                reporthook=t.update_to
            )
