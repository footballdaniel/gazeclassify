from __future__ import annotations

from src.persistence import Persistence, SaveData


class DescriptiveStats:

    def __init__(self, persistence: Persistence, logger) -> None:
        self.persistence = persistence
        self.logger = logger

    def number_frames_analyzed(self, df):
        frames_rated_by_algorithm_count = df.loc[df.Rater == "Algorithm", : ].groupby(['Trial', 'Frame']).count().shape[0]

        self.logger(f"Number of frames rated by algorithm: {frames_rated_by_algorithm_count}")
        self.persistence.add_result(SaveData.create(str(frames_rated_by_algorithm_count), "number_frames_analyzed"))

    def number_trials_analyzed(self, df):
        trials_rated_by_algorithm_count = df.loc[df.Rater == "Algorithm", : ].groupby(['Trial']).count().shape[0]
        trials_rated_by_algorithm_and_all_three_raters = df.groupby(['Trial', 'Frame']).filter(lambda x: x['Rater'].nunique() == 4).groupby(['Trial']).count().shape[0]

        if trials_rated_by_algorithm_and_all_three_raters != trials_rated_by_algorithm_count:
            self.logger(f"Number of trials rated by algorithm and all three raters is not equal to the number of trials rated by algorithm: {trials_rated_by_algorithm_and_all_three_raters} vs {trials_rated_by_algorithm_count}")
