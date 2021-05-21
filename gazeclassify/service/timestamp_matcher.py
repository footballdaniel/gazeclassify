from dataclasses import dataclass, field
from typing import List


@dataclass
class TimestampMatcher:
    baseline_timestamps: List[float]
    to_be_matched_data: List[float]
    matched_timestamps: List[float] = field(default_factory=list)
    _search_index: int = 0

    def match_to_base_frame_rate(self, data: List[float]) -> List[float]:
        for current_baseline_timestamp in self.baseline_timestamps:
            self._match_data_to_baseline_index(current_baseline_timestamp, data)
        return self.matched_timestamps

    def _match_data_to_baseline_index(self, current_baseline_timestamp: float, data: List[float]) -> None:
        if self._is_data_timestamp_higher_than(current_baseline_timestamp):
            self.matched_timestamps.append(data[self._search_index])
        else:
            if self._has_not_ended(data):
                while (current_baseline_timestamp > self.to_be_matched_data[self._search_index]) & (
                        self._search_index < len(data) - 1):
                    self._search_index += 1

                self.matched_timestamps.append(data[self._search_index])
            else:
                self.matched_timestamps.append(data[-1])

    def _has_not_ended(self, data: List[float]) -> bool:
        has_not_ended = self._search_index < len(data) - 1
        return has_not_ended

    def _is_data_timestamp_higher_than(self, current_baseline_timestamp: float) -> bool:
        is_ahead = current_baseline_timestamp <= self.to_be_matched_data[self._search_index]
        return is_ahead