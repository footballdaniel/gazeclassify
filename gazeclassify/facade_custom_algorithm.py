from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation, example_trial
from gazeclassify.classifier.semantic import CustomSegmentation

analysis = Analysis()

PupilLoader(analysis).from_trial_folder(example_trial())

CustomSegmentation(analysis,)
SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
analysis.export_video()

analysis.pieplot("pieplot.png")


