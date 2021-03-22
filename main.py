from dataclasses import dataclass

# from pixellib.instance import instance_segmentation  # type: ignore
import numpy as np

from gazeclassify.serializer.pupil_repository import PupilInvisibleRepository
from gazeclassify.serializer.pupil_serializer import PupilDataSerializer
from gazeclassify.utils import memory_logging

readable = PupilInvisibleRepository().load_capture("gazeclassify/tests/data/")
serializer = PupilDataSerializer().deserialize(readable)
