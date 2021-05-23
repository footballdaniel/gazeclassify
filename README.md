# GazeClassify
PiPy package to algorithmically analyze eye-tracking data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/footballdaniel/gazeclassify/blob/main/colab.ipynb)
[![Test status](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml/badge.svg)](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml)

---

### What is GazeClassify?
GazeClassify is a package do facilitate the analysis of eye-tracking data. Anyone can analyze gaze data online with less than 10 lines of code. GazeClassify provides a way to automatize and standardize eye-tracking analysis.

Gazeclassify currently supports only eye tracking data from the PupilInvisible eye tracker.
```python
from gazeclassify.classifier.instance import InstanceSegmentation
from gazeclassify.classifier.semantic import SemanticSegmentation
from gazeclassify.eyetracker.pupilinvisible import PupilInvisibleLoader
from gazeclassify.service.analysis import Analysis

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data/trial")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
```