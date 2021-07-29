from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation, example_trial

analysis = Analysis()

PupilLoader(analysis).from_recordings_folder(example_trial())

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
analysis.export_video()

analysis.pieplot("pieplot.png")


