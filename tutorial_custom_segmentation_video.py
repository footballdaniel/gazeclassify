from gazeclassify import Analysis, PupilLoader
from gazeclassify.classifier.semantic import CustomSegmentation

analysis = Analysis()

PupilLoader(analysis).from_recordings_folder(
    r"C:\Users\eyetracking\ownCloud\FGB_MLP_The_Eye_Of_The_Master (Projectfolder)\Data WP1\GazeClassify\data\participants\p000_example_data\exports\001")

CustomSegmentation(analysis, "mask_rcnn_model.041-0.951009.h5").classify(
    ["Mat", "Vault", "Trampoline", "Queue", "Jumper"])
analysis.save_to_csv(".")
analysis.export_video(".")
