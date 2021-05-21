from gazeclassify import Analysis, PupilInvisibleLoader, SemanticSegmentation, InstanceSegmentation

analysis = Analysis()

PupilInvisibleLoader(analysis).from_trial_folder("gazeclassify/example_data/trial")

SemanticSegmentation(analysis).classify("Human_Shape")
InstanceSegmentation(analysis).classify("Human_Joints")

analysis.save_to_json()
