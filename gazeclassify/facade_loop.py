import os

from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation

experimental_folder = "experiment"
trials = [f.path for f in os.scandir(experimental_folder) if f.is_dir()]

for trial in trials:
    analysis = Analysis()

    PupilLoader(analysis).from_recordings_folder(trial)

    SemanticSegmentation(analysis).classify("Human_Shape")
    InstanceSegmentation(analysis).classify("Human_Joints")

    analysis.save_to_csv(experimental_folder)
    analysis.export_video(experimental_folder)



