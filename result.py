from dataclasses import dataclass
from typing import List

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
    y_mean_last_seconds: float
    y_p05_last_seconds: float
    y_p95_last_seconds: float


# Global instance that callers can modify at runtime if desired
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
    y_p05 = float(np.percentile(y_tail, 5))
    y_p95 = float(np.percentile(y_tail, 95))

    return DatasetResult(
        y_mean_last_seconds=y_mean,
        y_p05_last_seconds=y_p05,
        y_p95_last_seconds=y_p95,
    )


def get_dataset_result(dataset_params: DerivedDatasetParameters) -> DatasetResult:
    return _get_dataset_result(dataset_params, RESULT_PARAMETERS)
