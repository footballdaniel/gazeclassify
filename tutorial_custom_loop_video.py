import os

from gazeclassify import Analysis, PupilLoader
from gazeclassify.classifier.semantic import CustomSegmentation

experimental_folder = "experiment"
trials = [f.path for f in os.scandir(experimental_folder) if f.is_dir()]

for trial in trials:
    analysis = Analysis()

    PupilLoader(analysis).from_recordings_folder(trial)

    CustomSegmentation(analysis, "mask_rcnn_model.041-0.951009.h5").classify(["BG", "Mat", "Vault", "Trampoline", "Queue", "Jumper"])

    analysis.save_to_csv(experimental_folder)
    analysis.export_video(experimental_folder)



