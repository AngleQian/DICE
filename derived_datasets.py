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
    x_seconds_end_dropped: int = 0
    vertical_scaling_factor: float = 1.0
    horizontal_scaling_factor: float = 1.0
    label_override: Optional[ExperimentLabels] = None
    description_suffix: Optional[str] = None

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
        # Base already includes original label and working_dir
        base = self.original_parameters.description
        if self.label_override is not None:
            base = f"{base} | as {self.label_override.value}"
        if self.description_suffix and len(self.description_suffix) > 0:
            base = f"{base} | {self.description_suffix}"
        return base


# Populate this list with the derived datasets you want to generate
DERIVED_DATASET_PARAMETERS: List[DerivedDatasetParameters] = []


def _slice_indices_for_drops(
    *, num_entries: int, seconds_start_drop: int, seconds_end_drop: int, interval_s: int
) -> tuple[int, int]:
    """
    Compute [start_idx, end_idx) to trim time series by wall-clock seconds.

    Semantics:
    - seconds_start_drop removes data from the beginning; we round UP to the next
      full frame so we never keep any portion of a requested drop (ceil).
    - seconds_end_drop removes data from the end; we convert seconds to a frame
      count and truncate that many frames from the tail (ceil).
    - Bounds are clamped so the resulting slice is valid: 0 <= start <= end <= N.
    """
    # Convert requested second-based drops into frame counts using the dataset sampling interval
    start_idx = int(np.ceil(seconds_start_drop / float(interval_s))) if seconds_start_drop > 0 else 0
    end_drop_count = int(np.ceil(seconds_end_drop / float(interval_s))) if seconds_end_drop > 0 else 0

    # Translate tail drop into an exclusive end index (Python slice uses end-exclusive)
    end_idx = max(0, num_entries - end_drop_count)

    # Clamp to valid range and ensure non-decreasing indices
    start_idx = min(start_idx, num_entries)
    end_idx = min(max(end_idx, start_idx), num_entries)
    return start_idx, end_idx


def build_derived_series(params: DerivedDatasetParameters) -> tuple[list[float], list[list[float]], str]:
    """
    Build the derived time axis and nested displacement-Y values from the original dataset,
    applying optional head/tail trimming and horizontal/vertical scaling.

    Notes:
    - Only displacement Y is considered for derived datasets.
    - Horizontal scaling multiplies time values; vertical scaling multiplies Y values (after unit conversion in analysis).
    """
    original = params.original_parameters
    dataset = load_dataset(original, disable_throwaway=False)

    xs = dataset.data_entry_xs_s
    ys_nested = dataset.get_displacement_ys_adjusted()

    interval = original.time_interval_per_image_s
    start_idx, end_idx = _slice_indices_for_drops(
        num_entries=len(ys_nested),
        seconds_start_drop=params.x_seconds_start_dropped,
        seconds_end_drop=params.x_seconds_end_dropped,
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

    return xs, ys_nested, params.description


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

