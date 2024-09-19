from typing import List

import krippendorff
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score

from .persistence import Persistence, SaveData, SaveFigure


class Agreement:

    def __init__(self, persistence: Persistence, logger):
        self.persistence = persistence
        self.logger = logger

    def plot_pct_viewing_time(self, df, desired_order: List[str]):
        import matplotlib as mpl
        mpl.rcParams.update(
            {
                'font.size': 9
            }
        )
        
        ratings_per_frame = df.pivot(index=['Trial', 'Frame'], columns='Rater', values='Label')
        ratings_per_frame = ratings_per_frame.dropna()

        # TO GET WITHOUT OTHER:
        # ratings_per_frame_no_other = ratings_per_frame[~ratings_per_frame.isin(["Other"]).any(axis=1)]

        # results including "other"
        pct_algorithm = ratings_per_frame.Algorithm.value_counts(normalize=True) * 100
        pct_rater_1 = ratings_per_frame.Rater1.value_counts(normalize=True) * 100
        pct_rater_2 = ratings_per_frame.Rater2.value_counts(normalize=True) * 100
        pct_rater_3 = ratings_per_frame.Rater3.value_counts(normalize=True) * 100
        
        pct_algorithm = pct_algorithm.reindex(desired_order)
        pct_rater_1 = pct_rater_1.reindex(desired_order)
        pct_rater_2 = pct_rater_2.reindex(desired_order)
        pct_rater_3 = pct_rater_3.reindex(desired_order)


        LATEX_WIDTH_INCHES = 5.9
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (LATEX_WIDTH_INCHES, 3), sharey = True)

        pct_algorithm.plot(kind='bar', ax = axes[0], color = '#909090')
        pct_rater_1.plot(kind = 'bar', ax = axes[1], color = '#909090')
        pct_rater_2.plot(kind = 'bar', ax = axes[2], color = '#909090')
        pct_rater_3.plot(kind = 'bar', ax = axes[3], color = '#909090')

        axes[0].set_title('Algorithm')
        axes[1].set_title('Rater 1')
        axes[2].set_title('Rater 2')
        axes[3].set_title('Rater 3')

        plt.tight_layout()
        plt.subplots_adjust(left=0.1)

        axes[0].set_ylabel('Viewing time [pct]')

        for ax in axes:
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey')

        self.persistence.add_figure(SaveFigure(fig, "pct_viewing_time"))

    def individual_ratings(self, df_long):
        df = self.to_wide_and_drop_na(df_long)

        columns = ["Algorithm", "Rater1", "Rater2", "Rater3"]
        data = df[columns].values
        factors, _ = pd.factorize(data.ravel())
        factors = factors.reshape(data.shape)
        df = pd.DataFrame(factors, columns=columns)

        # calculate agreement cohens kappa
        kappa_r1_algorithm = cohen_kappa_score(df.Rater1, df.Algorithm)
        kappa_r2_algorithm = cohen_kappa_score(df.Rater2, df.Algorithm)
        kappa_r3_algorithm = cohen_kappa_score(df.Rater3, df.Algorithm)

        kappa_r1_r2 = cohen_kappa_score(df.Rater1, df.Rater2)
        kappa_r1_r3 = cohen_kappa_score(df.Rater1, df.Rater3)
        kappa_r2_r3 = cohen_kappa_score(df.Rater2, df.Rater3)

        self.logger(f"Kappa r1-r2: {kappa_r1_r2:.2f}")
        self.logger(f"Kappa r1-r3: {kappa_r1_r3:.2f}")
        self.logger(f"Kappa r2-r3: {kappa_r2_r3:.2f}")

        self.logger(f"Kappa r1-algorithm: {kappa_r1_algorithm:.2f}")
        self.logger(f"Kappa r2-algorithm: {kappa_r2_algorithm:.2f}")
        self.logger(f"Kappa r3-algorithm: {kappa_r3_algorithm:.2f}")

        self.persistence.add_result(SaveData(f"{kappa_r1_r2:.2f}", "Kappa agreement human raters 1 and 2"))
        self.persistence.add_result(SaveData(f"{kappa_r1_r3:.2f}", "Kappa agreement human raters 1 and 3"))
        self.persistence.add_result(SaveData(f"{kappa_r2_r3:.2f}", "Kappa agreement human raters 2 and 3"))

        self.persistence.add_result(SaveData(f"{kappa_r1_algorithm:.2f}", "Kappa agreement human rater 1 and algorithm"))
        self.persistence.add_result(SaveData(f"{kappa_r2_algorithm:.2f}", "Kappa agreement human rater 2 and algorithm"))
        self.persistence.add_result(SaveData(f"{kappa_r3_algorithm:.2f}", "Kappa agreement human rater 3 and algorithm"))


    def individual_ratings_no_other(self, df_long):
        df = self.to_wide_and_drop_na(df_long)

        # Delete all "other" labels
        df = df[~df.isin(["Other"]).any(axis=1)]

        columns = ["Algorithm", "Rater1", "Rater2", "Rater3"]
        data = df[columns].values
        factors, _ = pd.factorize(data.ravel())
        factors = factors.reshape(data.shape)
        df = pd.DataFrame(factors, columns=columns)

        # calculate agreement cohens kappa
        kappa_r1_algorithm = cohen_kappa_score(df.Rater1, df.Algorithm)
        kappa_r2_algorithm = cohen_kappa_score(df.Rater2, df.Algorithm)
        kappa_r3_algorithm = cohen_kappa_score(df.Rater3, df.Algorithm)

        kappa_r1_r2 = cohen_kappa_score(df.Rater1, df.Rater2)
        kappa_r1_r3 = cohen_kappa_score(df.Rater1, df.Rater3)
        kappa_r2_r3 = cohen_kappa_score(df.Rater2, df.Rater3)

        self.logger(f"Kappa r1-r2: {kappa_r1_r2:.2f}")
        self.logger(f"Kappa r1-r3: {kappa_r1_r3:.2f}")
        self.logger(f"Kappa r2-r3: {kappa_r2_r3:.2f}")

        self.logger(f"Kappa r1-algorithm: {kappa_r1_algorithm:.2f}")
        self.logger(f"Kappa r2-algorithm: {kappa_r2_algorithm:.2f}")
        self.logger(f"Kappa r3-algorithm: {kappa_r3_algorithm:.2f}")

        self.persistence.add_result(SaveData(f"{kappa_r1_r2:.2f}", "Kappa agreement human raters 1 and 2 (no other)"))
        self.persistence.add_result(SaveData(f"{kappa_r1_r3:.2f}", "Kappa agreement human raters 1 and 3 (no other)"))
        self.persistence.add_result(SaveData(f"{kappa_r2_r3:.2f}", "Kappa agreement human raters 2 and 3 (no other)"))

        self.persistence.add_result(SaveData(f"{kappa_r1_algorithm:.2f}", "Kappa agreement human rater 1 and algorithm (no other)"))
        self.persistence.add_result(SaveData(f"{kappa_r2_algorithm:.2f}", "Kappa agreement human rater 2 and algorithm (no other)"))
        self.persistence.add_result(SaveData(f"{kappa_r3_algorithm:.2f}", "Kappa agreement human rater 3 and algorithm (no other)"))
    


    def pct_of_frames_agreed(self, df_long):
        df = self.to_wide_and_drop_na(df_long)
        df = self.human_agreement(df)
        pct_of_frames_3_humans_agreed = df.loc[df.Agreement == 3].shape[0] / df.shape[0] * 100
        pct_of_frames_at_least_2_humans_agreed = df.loc[df.Agreement >= 2].shape[0] / df.shape[0] * 100
        pct_of_frames_3_humans_agreed = f"{pct_of_frames_3_humans_agreed:.2f}%"
        pct_of_frames_at_least_2_humans_agreed = f"{pct_of_frames_at_least_2_humans_agreed:.2f}%"
        number_of_frames_without_perfect_agreement = df.loc[df.Agreement < 3].shape[0]

        self.logger('Pct of frames all 3 people rated the same: ' + pct_of_frames_3_humans_agreed)
        self.logger('Pct of at least 2 people rated the same: ' + pct_of_frames_at_least_2_humans_agreed)
        self.logger('Number of frames without perfect agreement: ' + str(number_of_frames_without_perfect_agreement))

        self.persistence.add_result(SaveData(pct_of_frames_3_humans_agreed, "pct_of_frames_3_humans_agreed"))
        self.persistence.add_result(SaveData.create(number_of_frames_without_perfect_agreement, "number_of_frames_without_perfect_human_agreement"))
        self.persistence.add_result(SaveData(pct_of_frames_at_least_2_humans_agreed, "pct_of_frames_at_least_2_humans_agreed"))

    def to_wide_and_drop_na(self, df_long):
        df_wide = df_long.pivot(index=['Trial', 'Frame'], columns='Rater', values='Label')
        df = df_wide[~(df_wide.isna()).any(axis=1)]
        return df

    def human_agreement(self, df):
        raters = df.loc[:, [col for col in df.columns if "Rater" in col]]
        agreement = raters.apply(lambda x: x.value_counts().max(), axis=1)
        df = df.assign(Agreement = agreement)
        return df
    
    def viewing_time_agreement_for_3_raters(self, df_long):
        min_agreement = 3
        df = self.to_wide_and_drop_na(df_long)
        df = self.human_agreement(df)
        df = df.loc[df.Agreement >= min_agreement]

        raters = df.loc[:, [col for col in df.columns if "Rater" in col]]
        aoi = raters.apply(lambda x: x.value_counts().idxmax(), axis=1)
        df = df.assign(HumanMajorityVote = aoi)
        most_popular_aoi = df.HumanMajorityVote.value_counts().idxmax()
        number_of_frames_for_most_popular_aoi = df.HumanMajorityVote.value_counts().max()

        percent_of_each_aoi = df.HumanMajorityVote.value_counts(normalize=True) * 100
        
        self.logger(f'Most frequent AOI for at least {min_agreement} human raters was "{most_popular_aoi}" with {number_of_frames_for_most_popular_aoi} frames')        

        def savedata_for_each_aoi():
            for aoi, percent in percent_of_each_aoi.items():
                yield SaveData(f"{percent:.2f}%", f"Viewing time {min_agreement} human raters for {aoi} for")

        for save_data in savedata_for_each_aoi():
            self.persistence.add_result(save_data)

        
    def viewing_time_agreement_for_2_raters(self, df_long):
        min_agreement = 2
        df = self.to_wide_and_drop_na(df_long)
        df = self.human_agreement(df)
        df = df.loc[df.Agreement >= min_agreement]

        raters = df.loc[:, [col for col in df.columns if "Rater" in col]]
        aoi = raters.apply(lambda x: x.value_counts().idxmax(), axis=1)
        df = df.assign(HumanMajorityVote = aoi)
        most_popular_aoi = df.HumanMajorityVote.value_counts().idxmax()
        number_of_frames_for_most_popular_aoi = df.HumanMajorityVote.value_counts().max()
        self.logger(f'Most frequent AOI for at least {min_agreement} human raters was "{most_popular_aoi}" with {number_of_frames_for_most_popular_aoi} frames')        

        text = f"{most_popular_aoi} with {number_of_frames_for_most_popular_aoi} frames"
        self.persistence.add_result(SaveData(text, "most_popular_aoi_at_least_2_humans_agreed"))

    def krippendorff_humans(self, df):
        df = df.assign(LabelFactor=pd.factorize(df.Label)[0])
        ratings_per_frame = df.pivot(index=['Trial', 'Frame'], columns='Rater', values='LabelFactor')
        ratings_per_frame = ratings_per_frame.dropna()

        rater_1 = ratings_per_frame.Rater1.to_numpy()
        rater_2 = ratings_per_frame.Rater2.to_numpy()
        rater_3 = ratings_per_frame.Rater3.to_numpy() 

        data = [rater_1, rater_2, rater_3]
        alpha = Krippendorff.alpha(data)
        self.logger(f'Krippendorff alpha humnas: {alpha:.2f}')
        self.persistence.add_result(SaveData(f"{alpha:.2f}", "krippendorff_alpha_humans"))

    def krippendorff_all(self, df):
        df = df.assign(LabelFactor=pd.factorize(df.Label)[0])
        ratings_per_frame = df.pivot(index=['Trial', 'Frame'], columns='Rater', values='LabelFactor')
        ratings_per_frame = ratings_per_frame.dropna()

        rater_1 = ratings_per_frame.Rater1.to_numpy()
        rater_2 = ratings_per_frame.Rater2.to_numpy()
        rater_3 = ratings_per_frame.Rater3.to_numpy()
        algorithm = ratings_per_frame.Algorithm.to_numpy()

        data = [rater_1, rater_2, rater_3, algorithm]
        alpha = Krippendorff.alpha(data)
        self.logger(f'Krippendorff alpha all: {alpha:.2f}')
        self.persistence.add_result(SaveData(f"{alpha:.2f}", "krippendorff_alpha_all"))

    def krippendorff_all_no_others(self, df):
        factors, categories = pd.factorize(df.Label)
        df = df.assign(LabelFactor=factors)
        ratings_per_frame = df.pivot(index=['Trial', 'Frame'], columns='Rater', values='LabelFactor')
        ratings_per_frame = ratings_per_frame.dropna()

        others_index = categories.tolist().index("Other")
        ratings_per_frame_no_others = ratings_per_frame.loc[ratings_per_frame.Algorithm != others_index]

        rater_1 = ratings_per_frame_no_others.Rater1.to_numpy()
        rater_2 = ratings_per_frame_no_others.Rater2.to_numpy()
        rater_3 = ratings_per_frame_no_others.Rater3.to_numpy()
        algorithm = ratings_per_frame_no_others.Algorithm.to_numpy()

        data = [rater_1, rater_2, rater_3, algorithm]
        alpha = Krippendorff.alpha(data)
        self.logger(f'Krippendorff alpha all no others: {alpha:.2f}')
        self.persistence.add_result(SaveData(f"{alpha:.2f}", "krippendorff_alpha_all_no_others"))

class Krippendorff:

    def __init__(self, observations_per_rater: List[List[int]]):
        self.observations_per_rater = observations_per_rater

    @staticmethod
    def alpha(observations_per_rater: List[List[int]]):
        alpha = krippendorff.alpha(observations_per_rater, level_of_measurement='nominal')
        return alpha        
