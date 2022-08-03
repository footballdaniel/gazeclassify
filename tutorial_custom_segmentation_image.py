from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation, example_trial
from gazeclassify.classifier.semantic import CustomSegmentation
from gazeclassify.domain.dataset import DataRecord, GazeData
from gazeclassify.thirdparty.opencv_api import OpenCVReader
from gazeclassify.thirdparty.pixellib_api import PixellibCustomTensorflowClassifier

# Learnings: the infer config needs to take exactly the right number of classes in
# (all that it was trained on. Each entry is four bytes

# Working example with squirrels.
from matplotlib import image
frame = image.imread("squirrel_squirrel_butterfly.jpg")
classifier = PixellibCustomTensorflowClassifier("Nature_model_resnet101.h5", ["butterfly",  "squirrel"])
classifier.set_target()
classifier.classify_frame(frame)
record = DataRecord(None, None, GazeData(0.5, 0.5))
result = classifier.gaze_distance_to_object(record)

# Now for the real dataset:
analysis = Analysis()
PupilLoader(analysis).from_recordings_folder(example_trial())
CustomSegmentation(analysis, "Nature_model_resnet101.h5").classify(["butterfly",  "squirrel"])
analysis.save_to_csv()
analysis.export_video()


# from pixellib import custom_segmentation
# video_name = "video_1.mp4"
# model_name = "mask_rcnn_models_head/mask_rcnn_model.223-0.627243.h5"
# class_names = ["BG", "Head"]
# # Maybe see if i can run the classes first:
# segment_image = custom_segmentation()
# segment_image.inferConfig(num_classes=1, class_names=class_names)
# segment_image.load_model(model_name)
#
# segment_image.process_video(f"raw_input/{video_name}", frames_per_second=20, output_video_name=f"processed_video/{video_name}_head.mp4")



