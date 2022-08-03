import os
from dataclasses import dataclass
from pathlib import Path
import logging

class Find:

    @classmethod
    def recordings_in(cls, recording_folder: str, results_folder: str) -> None:
        cls.recording_folder = recording_folder
        cls.results_folder = results_folder

        if not Path(cls.recording_folder).exists:
            raise Exception(f"Recording folder not found at path: '{cls.recording_folder}'")
        if not Path(cls.results_folder).exists:
            raise Exception("Results folder not found at path: '{results_folder}'")

        export_targets = cls.search_recording_folder()

        if cls._no_results_folder(cls.results_folder):
            return list(export_targets.values())

        cls.omit_already_analysed_results(export_targets)

        logging.info(f"Found {len(export_targets)} recordings remaining to export")
        return list(export_targets.values())

    @classmethod
    def omit_already_analysed_results(cls, export_targets):

        recording_name = "asfds"
        already_registered_results = [f.path for f in os.scandir(cls.results_folder) if f.is_file()]
        for result_file in already_registered_results:
            if recording_name in result_file:
                export_targets.pop(recording_name)

    @classmethod
    def search_recording_folder(cls):
        all_recordings = [f.path for f in os.scandir(cls.recording_folder) if f.is_dir()]
        export_targets = {}
        for recording_folder in all_recordings:
            recording_folder = Path(recording_folder)
            export_folder = recording_folder.joinpath("exports")

            recording = cls._last_export_from(export_folder)
            if not recording:
                continue

            recording_name = Path(export_folder).parent.name
            export_targets[recording_name] = recording
        return export_targets

    @classmethod
    def _last_export_from(self, export_path: Path) -> str:
        if export_path.is_dir():
            first_export = sorted(export_path.iterdir(), key=lambda x: x.name)

            if len(first_export) > 0:
                first_export = first_export.pop()
            else:
                logging.warning(f"No exported recordings found for recording '{recording_folder}'")
                return None
        else:
            logging.warning(f"Recordings folder not found at path: '{recording_folder}'")
            return None

        path = str(export_path.joinpath(first_export.name))

        return path

    @classmethod
    def _no_results_folder(cls, results_folder: str) -> bool:
        return Path(cls.results_folder).exists



