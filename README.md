# GazeClassify
PiPy package to algorithmically annotate eye-tracking data. 

Special thanks to [Pixellib](https://pixellib.readthedocs.io/en/latest/) for providing algorithms and to [PySport](https://github.com/PySport/kloppy) for inspiration with the domain model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/footballdaniel/gazeclassify/blob/main/colab.ipynb)
[![Test status](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml/badge.svg)](https://github.com/footballdaniel/gazeclassify/actions/workflows/test.yml)
[![Downloads](https://pepy.tech/badge/gazeclassify)](https://pepy.tech/project/gazeclassify)
[![Downloads](https://pepy.tech/badge/gazeclassify/week)](https://pepy.tech/project/gazeclassify)

---
### What is GazeClassify?
 GazeClassify provides automatized and standardized eye-tracking annotation. Anyone can analyze gaze data online with less than 10 lines of code. 

![Result_image](gazeclassify/example_data/result_composite.jpg)

Exported `csv` will contain distance from gaze (red circle) to human joints (left image) and human shapes (right image) for each frame.

| frame number 	| classifier name 	| gaze_distance [pixel] 	| person_id 	| joint name 	|
|--------------	|-----------------	|-----------------------	|-----------	|------------	|
| 0            	| Human_Joints    	| 79                    	| 0         	| Neck       	|
| ...          	| ...             	| ...                   	| ...       	| ...        	|
| 0            	| Human_Shape     	| 0                     	| None      	| None       	|
| ...          	| ...             	| ...                   	| ...       	| ...        	|

### Run on example data

```python
from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation
from gazeclassify import example_trial

analysis = Analysis()

PupilLoader(analysis).from_trial_folder(example_trial())

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
```

### Run on your own data
Capture eye tracking data from a Pupil eye tracker. Then, [export the data](https://docs.pupil-labs.com/core/#_8-export-data) using Pupil software. You will get a folder with the exported world video and the gaze timestamps. Finally, let gazeclassify analyze the exported data:

```python
from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation

analysis = Analysis()

PupilLoader(analysis).from_trial_folder("path/to/your/folder_with_exported_data/")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
```

### Citation
Please [cite this paper](https://dl.acm.org/doi/10.1145/3450341.3458886) in your publications if GazeClassify helps your research.

```
@inproceedings{10.1145/3450341.3458886,
  author = {M\"{u}ller, Daniel and Mann, David},
  title = {Algorithmic Gaze Classification for Mobile Eye-Tracking},
  year = {2021},
  booktitle = {ACM Symposium on Eye Tracking Research and Applications}
}
```

