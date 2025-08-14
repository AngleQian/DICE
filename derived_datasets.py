from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from parameters import (
    DatasetParameters,
    ExperimentLabels,
    find_dataset_parameters,
)
from analysis import (
    PROJECT_ROOT,
    load_dataset,
    plot_dataset,
)


@dataclass
class DerivedDatasetParameters:
    working_dir: str
    x_seconds_start_dropped: int = 0
    # Keep this many seconds after the initial drop. If None or <= 0, keep all remaining
    x_seconds_keep: Optional[int] = None
    vertical_scaling_factor: float = 1.0
    horizontal_scaling_factor: float = 1.0
    label_override: Optional[ExperimentLabels] = None
    # Noise suppression: centered moving average window (points) and compression factor in [0,1]
    moving_average_window_points: int = 100
    offset_compression_factor: float = 1.0  # 1.0 = no suppression; 0.0 = fully average
    # Data augmentation: if enabled, replace the series with a synthetic line-fit + noise
    augment_data: bool = False
    augmentation_noise_scale: float = 1.0  # Multiplier on estimated noise std

    @property
    def original_parameters(self) -> DatasetParameters:
        return find_dataset_parameters(labels=None, working_dirs=[self.working_dir])[0]

    @property
    def original_label(self) -> ExperimentLabels:
        return self.original_parameters.label

    @property
    def effective_label(self) -> ExperimentLabels:
        """Label used for grouping/sorting; prefers override when provided."""
        return self.label_override or self.original_label

    @property
    def description(self) -> str:
        # Show overridden label first if it exists, then original description
        if self.label_override is not None:
            return f"New label {self.label_override.value} | {self.original_parameters.description}"
        return self.original_parameters.description


# Populate this list with the derived datasets you want to generate
DERIVED_DATASET_PARAMETERS: List[DerivedDatasetParameters] = [
    # Liner 1 80p, -35 @ 5000
    # DerivedDatasetParameters(
    #     working_dir="0726Midnight_WorkingDir",
    #     x_seconds_keep=15000,
    #     vertical_scaling_factor=1.17,
    #     offset_compression_factor=0.7,
    # ),
    # DerivedDatasetParameters(
    #     working_dir="0620Midnight_WorkingDir",
    #     x_seconds_keep=15000,
    #     vertical_scaling_factor=1.0,
    #     offset_compression_factor=0.9,
    # ),
    DerivedDatasetParameters(
        working_dir="0621Midnight_WorkingDir",
        x_seconds_keep=15000,
        vertical_scaling_factor=1.2,
        offset_compression_factor=0.8,
    ),
    # Liner 1 100p, -47 @ 5500
    # Liner 1 120p, -55 @ 6000

    # Liner 2 80p, -23 @ 3000
    # Liner 2 100p, -26 @ 3500
    # Liner 2 120p, -30 @ 4000
]


def _slice_indices_for_window(
    *, num_entries: int, seconds_start_drop: int, seconds_keep: Optional[int], interval_s: int
) -> tuple[int, int]:
    """
    Compute [start_idx, end_idx) to trim time series by wall-clock seconds using
    a head drop plus a fixed-length keep window.

    Semantics:
    - seconds_start_drop removes data from the beginning; we round UP to the next
      full frame so we never keep any portion of a requested drop (ceil).
    - seconds_keep defines the duration to keep starting at start_idx; we round UP
      to ensure the requested duration is fully covered. If None or <= 0, keep all
      remaining frames after the start drop.
    - Bounds are clamped so the resulting slice is valid: 0 <= start <= end <= N.
    """

    # Calculate start index from the seconds to drop at the head
    start_idx = int(np.ceil(seconds_start_drop / float(interval_s))) if seconds_start_drop > 0 else 0

    # If keep window is provided and positive, compute an exclusive end index
    if seconds_keep is not None and seconds_keep > 0:
        keep_count = int(np.ceil(seconds_keep / float(interval_s)))
        end_idx = start_idx + keep_count
    else:
        end_idx = num_entries

    # Clamp to valid range and ensure non-decreasing indices
    start_idx = min(start_idx, num_entries)
    end_idx = min(max(end_idx, start_idx), num_entries)
    return start_idx, end_idx


def _moving_average_centered(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average using edge padding. Returns array of same length."""
    if window <= 1:
        return values.copy()
    # Split padding for even/odd window sizes to keep centered alignment
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _apply_noise_suppression(means: np.ndarray, window: int, compress: float) -> np.ndarray:
    """
    Compress deviations from the centered moving average by factor `compress`.
    new = avg + compress * (orig - avg)
    - compress=1.0 -> no change; compress=0.0 -> fully smoothed to the moving average
    """
    if compress >= 1.0 or window <= 1:
        return means
    avg = _moving_average_centered(means, window)
    return avg + compress * (means - avg)


def build_derived_series(params: DerivedDatasetParameters) -> tuple[list[float], list[list[float]], str]:
    """
    Build the derived time axis and nested displacement-Y values from the original dataset,
    applying optional head/tail trimming, horizontal/vertical scaling, and noise suppression.

    Notes:
    - Only displacement Y is considered for derived datasets.
    - Horizontal scaling multiplies time values; vertical scaling multiplies Y values (after unit conversion in analysis).
    - Noise suppression operates on the per-image mean Y series.
    """
    original = params.original_parameters
    dataset = load_dataset(original, disable_throwaway=False)

    xs = dataset.data_entry_xs_s
    ys_nested = dataset.get_displacement_ys_adjusted()

    interval = original.time_interval_per_image_s
    start_idx, end_idx = _slice_indices_for_window(
        num_entries=len(ys_nested),
        seconds_start_drop=params.x_seconds_start_dropped,
        seconds_keep=params.x_seconds_keep,
        interval_s=interval,
    )

    # Apply trims using calculated slice indices
    xs = xs[start_idx:end_idx]
    ys_nested = ys_nested[start_idx:end_idx]

    # Apply horizontal (time) scaling if requested
    if params.horizontal_scaling_factor != 1.0:
        xs = [x * params.horizontal_scaling_factor for x in xs]

    # Apply vertical (measurement) scaling if requested
    if params.vertical_scaling_factor != 1.0:
        ys_nested = [[y * params.vertical_scaling_factor for y in sub] for sub in ys_nested]

    # Compute per-image means
    means = np.array([float(np.mean(sub)) for sub in ys_nested], dtype=float)

    # Always apply noise suppression first
    means_suppressed = _apply_noise_suppression(
        means,
        window=params.moving_average_window_points,
        compress=params.offset_compression_factor,
    )

    if params.augment_data:
        # Fit a line to the suppressed series and simulate noise based on recent suppressed residuals
        x_arr = np.array(xs, dtype=float)
        if len(x_arr) >= 2 and np.std(x_arr) > 0:
            slope, intercept = np.polyfit(x_arr, means_suppressed, 1)
            fitted = slope * x_arr + intercept
            residuals = means_suppressed - fitted
        else:
            # Degenerate case: not enough points or zero time variance; treat as constant
            slope, intercept = 0.0, float(means_suppressed[-1] if len(means_suppressed) > 0 else 0.0)
            fitted = np.full_like(means_suppressed, intercept)
            residuals = means_suppressed - fitted

        window_pts = params.moving_average_window_points if params.moving_average_window_points > 1 else len(residuals)
        window_pts = min(window_pts, len(residuals)) if len(residuals) > 0 else 0
        noise_std = float(np.std(residuals[-window_pts:])) if window_pts > 0 else 0.0

        rng = np.random.default_rng()
        simulated_noise = rng.normal(loc=0.0, scale=noise_std * params.augmentation_noise_scale, size=len(x_arr))
        augmented = fitted + simulated_noise
        ys_nested_adjusted = [[float(v)] for v in augmented.tolist()]
    else:
        # No augmentation; use suppressed series as-is
        ys_nested_adjusted = [[val] for val in means_suppressed.tolist()]

    return xs, ys_nested_adjusted, params.description


def export_derived_y_plots(derived_params: List[DerivedDatasetParameters]) -> None:
    if len(derived_params) == 0:
        print("No derived dataset parameters specified; nothing to export.")
        return

    # Sort using label override when present
    derived_params_sorted = sorted(derived_params, key=lambda p: p.effective_label.value)

    output_file_y = PROJECT_ROOT / "derived_analysis_output_displacement_y.pdf"

    plots = []
    for dp in derived_params_sorted:
        xs, nested_ys, desc = build_derived_series(dp)
        fig = plot_dataset(xs, nested_ys, desc, "derived displacement y")
        plots.append(fig)

    with PdfPages(output_file_y) as pdf:
        for fig in plots:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Exported derived Y displacement plots to: {output_file_y}")


if __name__ == "__main__":
    export_derived_y_plots(DERIVED_DATASET_PARAMETERS)

