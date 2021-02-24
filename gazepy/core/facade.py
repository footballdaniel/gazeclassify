from gazepy.core.models import Dataset, Metadata
from gazepy.core.data_loader import PupilDataDeserializer


def load_from_pupil_invisible(path: str) -> Dataset:
    data = PupilDataDeserializer().load_from_export_folder(path)
    print("Has Loaded Data")

    # Require return dataset
    metadata = Metadata("str")
    dataset = Dataset(metadata)
    return dataset
