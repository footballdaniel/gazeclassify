import os
import urllib.request
import pathlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelDownload:
    _model_name: str = "mask_rcnn_coco.h5"

    @property
    def download_url(self) -> str:
        return "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"

    def download(self) -> Path:
        base_dir = self._get_local_folder()
        model_file = self._download_model(base_dir)
        return model_file

    def _get_local_folder(self) -> str:
        base_dir = os.path.expanduser("~/gazeclassify_data/")
        pathlib.Path(base_dir).mkdir(exist_ok=True, parents=True)
        return base_dir

    def _download_model(
        self, base_dir: str, model_name: str = "mask_rcnn_coco.h5"
    ) -> Path:
        model_file = self._download_location(base_dir, model_name)
        self._download_if_not_existing(model_file)
        return model_file

    def _download_location(self, base_dir: str, model_name: str) -> Path:
        model_file = pathlib.Path(base_dir).joinpath(model_name)
        return model_file

    def _download_if_not_existing(self, model_file: Path) -> None:
        if not pathlib.Path.exists(model_file):
            urllib.request.urlretrieve(
                self.download_url,
                model_file,
            )
