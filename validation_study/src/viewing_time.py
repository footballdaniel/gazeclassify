import logging
from src.persistence import Persistence, SaveData


class ViewingTime:

    def __init__(self, persistence: Persistence, logger):
        self.persistence = persistence
        self.logger = logger


    def to_wide_and_drop_na(self, df_long):
        df_wide = df_long.pivot(index=['Trial', 'Frame'], columns='Rater', values='Label')
        df = df_wide[~(df_wide.isna()).any(axis=1)]
        return df


    def algorithm(self, df_long):
        df = self.to_wide_and_drop_na(df_long)
  
        algorithm = df.Algorithm.value_counts(normalize=True) * 100

        for aoi, percent in algorithm.items():
            self.persistence.add_result(SaveData(f"{percent:.2f}%", f"Viewing time Algorithm for {aoi}"))
            self.logger(f'Viewing time Algorithm for {aoi}: {percent:.2f}%')