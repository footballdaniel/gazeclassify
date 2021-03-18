from __future__ import annotations

from typing import Dict, List

from gazeclassify.serializer.pupil_serializer import TimestampMatcher


class Builder:

    def timestamps_dict(self) -> Builder:
        return self

    def with_same_length(self) -> Dict[str, List[float]]:
        data_same_length = {
            'baseline': [1., 2., 3.],
            'to be matched': [1., 3., 3.]
        }
        return data_same_length


class TestTimestampMatcher:

    def test_import_timestamp_from_dict_results_in_correct_assignment(self) -> None:
        data = Builder().timestamps_dict().with_same_length()
        matcher = TimestampMatcher(data['baseline'], data['to be matched'])
        number_world_timestamps = len(matcher.baseline_timestamps)
        number_gaze_timestamps = len(matcher.to_be_matched)
        assert number_gaze_timestamps == number_world_timestamps

    def test_matching_returns_length_of_baseline(self) -> None:
        data = Builder().timestamps_dict().with_same_length()
        matcher = TimestampMatcher(data['baseline'], data['to be matched'])
        result = matcher.match_to_baseline([100., 200., 300.])
        baseline_length = len(data['baseline'])
        result_length = len(result)
        assert baseline_length == result_length

    # def test_matching_returns_same_value_if_baseline_is_lower(self) -> None:
    #     data = Builder().timestamps_dict().with_same_length()
    #     matcher = TimestampMatcher(data['baseline'], data['to be matched'])
    #     result = matcher.match_to_baseline([100., 200., 300.])
    #     assert result == [100., 100., 300.]

