from __future__ import annotations
import json
import os
import docx

from matplotlib import pyplot as plt

class SaveTable:

    def __init__(self, data: docx.Document, tag: str) -> None:
        self._data = data
        self._tag = tag

    @property
    def data(self) -> str:
        return self._data

    @property
    def tag(self) -> str:
        return self._tag


class SaveData:

    @staticmethod
    def create(data: str, tag: str) -> SaveData:
        return SaveData(data, tag)

    def __init__(self, data: str, tag: str) -> None:
        self._data = data
        self._tag = tag

    @property
    def data(self) -> str:
        return self._data

    @property
    def tag(self) -> str:
        return self._tag


class SaveFigure:

    def __init__(self, data: plt.Figure, tag: str) -> None:
        self._data = data
        self._tag = tag

    @property
    def data(self) -> str:
        return self._data

    @property
    def tag(self) -> str:
        return self._tag


class Persistence:

    def __init__(self, folder: str, textformat: str = "json", plotformat: str = "png") -> None:
        self.folder = folder
        self.format = textformat
        self.plotformat = plotformat
        self.savedata = {}
        self.saveplots = {}
        self.savetables = {}
        
        if not os.path.exists(folder):    
            os.makedirs(self.folder)

    def add_result(self, content: SaveData) -> None:
        self.savedata[content.tag] = content.data

    def add_figure(self, content: SaveFigure) -> None:
        self.saveplots[content.tag] = content

    def add_table(self, content: SaveTable) -> None:
        self.savetables[content.tag] = content.data

    def save(self) -> None:
        for tag, data in self.savetables.items():
            data.save(os.path.join(self.folder, tag) + ".docx")

        for name, plot in self.saveplots.items():
            plot.data.savefig(os.path.join(self.folder, name + "." + self.plotformat))

        if self.format == "json":
            with open(f"{self.folder}/results.json", "w") as f:
                json.dump(self.savedata, f, indent=4)
        else:
            for tag, data in self.savedata.items():
                with open(f"{self.folder}/{tag}.csv", "w") as f:
                    f.write(data)




