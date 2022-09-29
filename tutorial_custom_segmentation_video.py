from gazeclassify import Analysis, PupilLoader
from gazeclassify.classifier.semantic import CustomSegmentation

analysis = Analysis()

PupilLoader(analysis).from_recordings_folder(
    r"C:\Path\to\file")

CustomSegmentation(analysis, "mask_rcnn_model.041-0.951009.h5").classify(
    ["Mat", "Vault", "Trampoline", "Queue", "Jumper"])
analysis.save_to_csv("./Results")
analysis.export_video("./Results")
