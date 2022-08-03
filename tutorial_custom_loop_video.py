import logging

from gazeclassify import Analysis, PupilLoader, Find
from gazeclassify.classifier.semantic import CustomSegmentation

recordings_folder = "../../../data_link/participants/"
model_file = "../../../data_link/models/mask_rcnn_model_all_AOIs.h5"
results_folder = "../../../data_link/results/"

all_recordings = Find.recordings_in(recordings_folder, results_folder)

for recording in all_recordings:
    analysis = Analysis()
    PupilLoader(analysis).from_recordings_folder(recording)
    CustomSegmentation(analysis, model_file).classify(["Mat", "Vault", "Trampoline", "Queue", "Jumper"])
    analysis.save_to_csv(results_folder)

logging.info("Done!")
