import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, BinaryIO, List, Any, cast

from gazeclassify.domain.dataset import Dataset
from gazeclassify.domain.results import FrameResult, InstanceClassification


class Serializer(ABC):

    @abstractmethod
    def deserialize(self, gaze_data: Dict[str, BinaryIO], video_metadata: Dict[str, str]) -> Dataset:
        ...

    @abstractmethod
    def serialize(self) -> Tuple[str, str]:
        raise NotImplementedError


class CSVSerializer:
    def encode(self, data: List[FrameResult], filename: Path) -> None:
        dict_data = self._frame_result_to_dict(data)
        csv_columns = ["frame", "name",  "distance", "person_id", "joint"]
        try:
            with open(str(filename), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for values in dict_data:
                    writer.writerow(values)
        except IOError:
            logging.error("Could not write to csv file, error in serializing results. Falling back to json-export")
            JsonSerializer().encode(data, filename)

        print("finished")

    def _frame_result_to_dict(self, data: List[FrameResult]) -> List[Dict[str, str]]:
        dict_data = []
        for result_idx, result in enumerate(data):
            for classification in data[result_idx].classifications:
                dict: Dict[str, str] = {}
                dict["frame"] = str(data[result_idx].frame_id)
                dict["name"] = data[result_idx].name
                dict["person_id"] = ""
                dict["joint"] = ""

                if classification.distance == None:
                    dict["distance"] = ""
                else:
                    dict["distance"] = str(int(classification.distance))  # type: ignore
                try:
                    instance = cast(InstanceClassification, classification)
                    dict["person_id"] = str(instance.person_id)
                    dict["joint"] = instance.joint
                except:
                    pass
                dict_data.append(dict)
        return dict_data


class JsonSerializer:
    def encode(self, data: object, filename: Path) -> None:
        with open(str(filename), "w") as write_file:
            json.dump(
                data,
                write_file,
                indent=4,
                sort_keys=False,
                cls=ClassesToDictEncoder
            )


class ClassesToDictEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Dict[Any, Any]:
        return obj.__dict__