from dataclasses import dataclass


@dataclass
class Metadata:
    recording_name: str


@dataclass
class Dataset:
    metadata: Metadata
