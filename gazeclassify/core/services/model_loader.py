import os.path
from dataclasses import dataclass
from urllib import request


@dataclass
class ModelLoader:
    model_url: str = ""
    data_path: str = "~/gazeclassify_data/"

    def _data_path_in_home_directory(self) -> str:
        return os.path.expanduser(self.data_path)

    def download_if_not_available(self, model_file: str) -> None:
        file_path = self._data_path_in_home_directory() + model_file
        if not os.path.exists(file_path):
            self._download_file(model_file)
        self.path = file_path

    def _download_file(self, model_file: str) -> None:
        remote_url = self.model_url
        local_file = self._data_path_in_home_directory() + model_file
        request.urlretrieve(remote_url, local_file)