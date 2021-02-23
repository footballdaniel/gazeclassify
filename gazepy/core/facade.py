from dataclasses import dataclass, fields
from gazepy.core.models import Dataset, Metadata
from gazepy.core.data_loader import PupilDataLoader


def load_from_pupil_invisible(self, path: str) -> Dataset:
    data = PupilDataLoader().load_from_export_folder(path)
    print("Has Loaded Data")

    # Require return dataset
    metadata = Metadata("str")
    dataset = Dataset(metadata)
    return dataset
