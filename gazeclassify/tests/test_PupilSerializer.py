from __future__ import annotations

from typing import Dict, List

from gazeclassify.service.timestamp_matcher import TimestampMatcher


class Builder:

    def timestamps_dict(self) -> Builder:
        return self

    def with_same_length(self) -> Dict[str, List[float]]:
        data_same_length = {
            'baseline': [1., 2., 3.],
            'to be matched': [1., 2., 3.]
        }
        return data_same_length

    def with_baseline_starting_lower(self) -> Dict[str, List[float]]:
        data_baseline_starts_lower = {
            'baseline': [1., 2., 3.],
            'to be matched': [2., 3., 4.]
        }
        return data_baseline_starts_lower

    def with_baseline_starting_higher(self) -> Dict[str, List[float]]:
        data_baseline_starts_lower_and_ending_lower = {
            'baseline': [2., 3., 4.],
            'to be matched': [1., 2., 3.]
        }
        return data_baseline_starts_lower_and_ending_lower

    def with_baseline_longer(self) -> Dict[str, List[float]]:
        data_baseline_longer = {
            'baseline': [1., 2., 3.],
            'to be matched': [2., 3.]
        }
        return data_baseline_longer

    def with_baseline_shorter(self) -> Dict[str, List[float]]:
        data_baseline_longer = {
            'baseline': [1., 2.],
            'to be matched': [2., 3., 4.]
        }
        return data_baseline_longer

    def with_to_be_matched_not_same(self) -> Dict[str, List[float]]:
        data_baseline_longer = {
            'baseline': [1., 2., 3.],
            'to be matched': [1.8, 1.95, 2.1]
        }
        return data_baseline_longer


class TestTimestampMatcher:
    class TestWhenArrayOfSameLength:
        def test_import_timestamp_from_dict_results_in_correct_assignment(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            number_world_timestamps = len(matcher.baseline_timestamps)
            number_gaze_timestamps = len(matcher.to_be_matched_data)
            assert number_gaze_timestamps == number_world_timestamps

        def test_matching_returns_result_with_same_length_of_baseline(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([100., 200., 300.])
            baseline_length = len(data['baseline'])
            result_length = len(result)
            assert baseline_length == result_length

        def test_matching_returns_same_start_value_when_first_index_the_same(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([100., 200., 300.])[0]
            assert result == 100

    class TestWhenBaselineStartsAtDifferentValue:
        def test_use_last_known_value_when_baseline_starts_higher(self) -> None:
            data = Builder().timestamps_dict().with_baseline_starting_higher()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([100., 200., 300.])
            assert result == [200., 300., 300.]

        def test_use_first_known_value_when_baseline_starts_lower(self) -> None:
            data = Builder().timestamps_dict().with_baseline_starting_lower()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([100., 200., 300.])
            assert result == [100., 100., 200.]

    class TestWhenBaselineHasMoreValues:
        def test_use_last_known_value_if_baseline_has_more_values(self) -> None:
            data = Builder().timestamps_dict().with_baseline_longer()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([200., 300.])
            assert result == [200., 200., 300.]

        def test_when_baseline_has_fewer_values(self) -> None:
            data = Builder().timestamps_dict().with_baseline_shorter()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([200., 300.])
            assert result == [200., 200.]

    class TestWhenBaselineValuesDontMatch:
        def test_baseline_values_are_slightly_lower_than_to_be_matched_values(self) -> None:
            data = Builder().timestamps_dict().with_to_be_matched_not_same()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_base_frame_rate([1.8, 1.95, 2.1])
            assert result == [1.8, 2.1, 2.1]
