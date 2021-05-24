# GazeClassify
PiPy package to algorithmically analyze eye-tracking data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/footballdaniel/gazeclassify/blob/main/colab.ipynb)
[![Test status](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml/badge.svg)](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml)

---

### What is GazeClassify?
GazeClassify is a package do facilitate the analysis of eye-tracking data. Anyone can analyze gaze data online with less than 10 lines of code. GazeClassify provides a way to automatize and standardize eye-tracking analysis.

Gazeclassify currently supports only eye tracking data from the PupilInvisible eye tracker.
```python
from gazeclassify import Analysis, PupilInvisibleLoader, SemanticSegmentation, InstanceSegmentation, example_trial

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder(example_trial())

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
```