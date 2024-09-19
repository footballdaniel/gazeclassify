import glob
import logging

import pandas as pd

from src.persistence import SaveData

class Preprocess:

    def __init__(self, persistence, logger):
        self.persistence = persistence
        self.logger = logger

    def load_data(self, file_pattern: str):
        # Load data from algorithmic tracking
        raterFiles = glob.glob(file_pattern + "P*.txt")
        df_algoFiles = (pd.read_csv(f, header=None) for f in raterFiles)
        df_algo = pd.concat(df_algoFiles, ignore_index=True, axis=0)
        # Load data from manual ratings
        raterFiles = glob.glob(file_pattern + "data_Rater*.csv")
        df_raterFiles = (pd.read_csv(f, header=0) for f in raterFiles)
        df_rater = pd.concat(df_raterFiles, ignore_index=True)

        unique_trials = len(df_rater.Trial.unique())
        unique_participants = df_rater['Trial'].str[:3]
        unique_participants = len(unique_participants.unique())

        self.logger(f"Unique trials: {unique_trials}")
        self.persistence.add_result(SaveData(f"{unique_trials}", "Unique trials"))

        self.logger("Unique Participants " + str(unique_participants))
        self.persistence.add_result(SaveData(f"{unique_participants}", "Unique participants"))

        # Only take the last judgement of each rater
        df_rater.drop_duplicates(subset=['Rater', 'Frame', 'Trial'], keep='last', inplace=True)
        df_algo.columns = ["Trial", "Label", "1", "2", "3", "4", "5", "6", "VisiblePoints", "7", "8"]
        df_algo["Frame"] = df_algo.groupby(['Trial']).cumcount()
        df_algo['Rater'] = 'Algorithm'
        df_algo["Trial"] = df_algo["Trial"].astype("string")
        df_algo["Frame"] = df_algo["Frame"].astype("string")
        df_algo["Label"] = df_algo["Label"].astype("string")
        df_rater["Frame"] = df_rater["Frame"].astype("string")
        df_rater["Trial"] = df_rater["Trial"].astype("string")
        df_rater["Label"] = df_rater["Label"].astype("string")
        # Rename the labels to match the AOI from the algorithmic approach
        df_algo['Label'] = df_algo['Label'].str.replace("Nose", "Head")
        df_algo['Label'] = df_algo['Label'].str.replace("Neck", "Chest")
        df_algo['Label'] = df_algo['Label'].str.replace("LElbow", "Left arm")
        df_algo['Label'] = df_algo['Label'].str.replace("RElbow", "Right arm")
        df_algo['Label'] = df_algo['Label'].str.replace("RKnee", "Right leg")
        df_algo['Label'] = df_algo['Label'].str.replace("LKnee", "Left leg")
        df_algo['Label'] = df_algo['Label'].str.replace("MidHip", "Pelvis")

        return pd.concat([df_algo, df_rater], keys=['Trial', 'Frame', 'Rater', 'Label']).reset_index(drop=True)
    
    def filter_rows(self, group):
        if group.shape[0] > 1:
            return group

    def delete_incomplete_ratings(self, df):
        filtered_df = df.groupby(['Trial', 'Frame']).apply(self.filter_rows)
        filtered_df.reset_index(drop=True, inplace=True)
        filtered_df.ffill(inplace=True)
        filtered_df = filtered_df[['Trial', 'Label', 'VisiblePoints', 'Frame', 'Rater']]
        return filtered_df
