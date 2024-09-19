from .human_agreement import Krippendorff


def test_Krippendorff_validation():

    rater_1_rated_5_items = [1, 0, 1, 1, 1]
    rater_2_rated_5_items = [1, 0, 1, 1, 1]
    rater_3_rated_5_items = [1, 0, 1, 1, 1]


    perfect_agreement = [
        rater_1_rated_5_items,
        rater_2_rated_5_items,
        rater_3_rated_5_items
    ]

    alpha = Krippendorff.alpha(perfect_agreement)
    assert alpha == 1.0