from __future__ import annotations

from typing import Dict, List

from gazeclassify.serializer.pupil_serializer import TimestampMatcher


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


class TestTimestampMatcher:

    class TestWhenArrayOfSameLength:
        def test_import_timestamp_from_dict_results_in_correct_assignment(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            number_world_timestamps = len(matcher.baseline_timestamps)
            number_gaze_timestamps = len(matcher.to_be_matched)
            assert number_gaze_timestamps == number_world_timestamps

        def test_matching_returns_result_with_same_length_of_baseline(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_baseline([100., 200., 300.])
            baseline_length = len(data['baseline'])
            result_length = len(result)
            assert baseline_length == result_length

        def test_matching_returns_same_start_value_when_first_index_the_same(self) -> None:
            data = Builder().timestamps_dict().with_same_length()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_baseline([100., 200., 300.])[0]
            assert result == 100

    class TestWhenBaselineStartsAtLowerValue:
        def test_use_first_known_value_as_baseline_value_is_lower(self) -> None:
            data = Builder().timestamps_dict().with_baseline_starting_lower()
            matcher = TimestampMatcher(data['baseline'], data['to be matched'])
            result = matcher.match_to_baseline([100., 200., 300.])
            assert result == [100., 100., 200.]

