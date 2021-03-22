from typing import Any

# from pixellib.instance import instance_segmentation  # type: ignore

from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer

readable = PupilInvisibleRepository().load_capture("gazeclassify/tests/data/")
serializer = PupilDataSerializer().deserialize(readable)




