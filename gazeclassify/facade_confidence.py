from gazeclassify import Analysis, PupilLoader, SemanticSegmentation, InstanceSegmentation, example_trial

analysis = Analysis()

PupilLoader(analysis).from_trial_folder(example_trial())

semantic = SemanticSegmentation(analysis)
semantic.minimal_confidence = 0.7
semantic.classify("Human_Shape")

instance = InstanceSegmentation(analysis)
instance.minimal_confidence = 0.7
instance.classify("Human_Joints")

analysis.save_to_csv()
analysis.export_video()
analysis.pieplot("pieplot.png")


