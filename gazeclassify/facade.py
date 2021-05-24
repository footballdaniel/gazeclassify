from gazeclassify import Analysis, PupilInvisibleLoader, SemanticSegmentation, InstanceSegmentation, example_trial

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder(example_trial())

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_csv()
