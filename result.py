from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from derived_datasets import (
    DerivedDatasetParameters,
    load_or_build_derived_series,
)


@dataclass
class ResultParameters:
    last_x_minutes: int

    @property
    def last_x_seconds(self) -> int:
        return self.last_x_minutes * 60



@dataclass
class DatasetResult:
    last_x_seconds_start: int
    last_x_seconds_end: int
    y_mean_last_seconds: float
    y_p05_last_seconds: float
    y_p95_last_seconds: float
    x_first_to_reach_y_mean_last_seconds: float

    @property
    def absolute_uncertainty(self) -> float:
        return max(self.y_p95_last_seconds - self.y_mean_last_seconds, self.y_mean_last_seconds - self.y_p05_last_seconds)
    
    @property
    def relative_uncertainty(self) -> float:
        return abs(self.absolute_uncertainty / self.y_mean_last_seconds) * 100


RESULT_PARAMETERS = ResultParameters(last_x_minutes=30)


def _get_dataset_result(
    dataset_params: DerivedDatasetParameters,
    result_params: ResultParameters,
) -> DatasetResult:
    xs, ys_nested, _desc = load_or_build_derived_series(dataset_params)

    if len(xs) == 0 or len(ys_nested) == 0:
        raise ValueError(f"No data to compute result {len(xs)=} {len(ys_nested)=} for {dataset_params.description=}")
    if len(xs) != len(ys_nested):
        raise ValueError(f"Mismatch between xs and ys_nested lengths {len(xs)=} {len(ys_nested)=} for {dataset_params.description=}")
    if max(xs) < result_params.last_x_seconds:
        raise ValueError(f"Last x seconds {result_params.last_x_seconds=} is greater than the maximum time {max(xs)=} for {dataset_params.description=}")

    # Convert nested Y values to a single scalar per time point (mean)
    y_series: List[float] = [float(np.mean(sub)) for sub in ys_nested]

    # Determine the starting index for the last X seconds window
    window_end_time = float(xs[-1])
    window_start_time = window_end_time - float(result_params.last_x_seconds)
    if result_params.last_x_seconds <= 0:
        window_start_time = float(xs[0])

    start_index = 0
    # Find the first index whose time is >= window_start_time
    for i, t in enumerate(xs):
        if float(t) >= window_start_time:
            start_index = i
            break

    y_tail = y_series[start_index:]

    y_mean = float(np.mean(y_tail))
    y_p05 = float(np.percentile(y_tail, 10))
    y_p95 = float(np.percentile(y_tail, 90))

    # Find the earliest time x based on a 100-point moving average window
    # whose average is less than or equal to the last-window mean. The
    # reported x is the time at the center of the first such window.
    x_first: Optional[float] = None
    window_size = 10
    n_points = len(y_series)
    if n_points > 0:
        effective_window = min(window_size, n_points)
        y_arr = np.array(y_series, dtype=float)
        cumsum = np.cumsum(y_arr)
        for start_idx in range(0, n_points - effective_window + 1):
            end_idx = start_idx + effective_window
            window_sum = cumsum[end_idx - 1] - (cumsum[start_idx - 1] if start_idx > 0 else 0.0)
            window_avg = window_sum / float(effective_window)
            if window_avg <= y_p95:
                center_idx = start_idx + (effective_window // 2)
                x_first = float(xs[center_idx])
                break
    if x_first is None:
        raise ValueError(f"No time found where the mean series first reaches or exceeds the mean value computed over the last-X-seconds window for {dataset_params.description=}")

    return DatasetResult(
        last_x_seconds_start=window_start_time,
        last_x_seconds_end=window_end_time,
        y_mean_last_seconds=y_mean,
        y_p05_last_seconds=y_p05,
        y_p95_last_seconds=y_p95,
        x_first_to_reach_y_mean_last_seconds=x_first,
    )


def get_dataset_result(dataset_params: DerivedDatasetParameters) -> DatasetResult:
    return _get_dataset_result(dataset_params, RESULT_PARAMETERS)
