from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import hashlib
import pickle
import json
import logging

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
from logging_utils import setup_logging

# Cache configuration for derived datasets
DERIVED_CACHE_DIR = PROJECT_ROOT / ".cache_derived"
DERIVED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_CACHE_VERSION = 1


def _params_dict(p: "DerivedDatasetParameters") -> dict:
    d = asdict(p)
    # Normalize Enum to string for stable hashing
    label_override = d.get("label_override")
    if label_override is not None:
        d["label_override"] = (
            p.label_override.value if isinstance(p.label_override, ExperimentLabels) else str(label_override)
        )
    return d


def _params_hash(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _cache_path_for_key(key: str) -> object:
    return DERIVED_CACHE_DIR / f"{key}.pkl"


def load_or_build_derived_series(params: "DerivedDatasetParameters") -> tuple[list[float], list[list[float]], str]:
    """
    Transparent loader with file cache. If cache exists for params, load it;
    otherwise build via `build_derived_series`, persist, and return.
    """
    DERIVED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    params_d = _params_dict(params)
    key = _params_hash(params_d)
    cache_path = _cache_path_for_key(key)

    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            if (
                isinstance(payload, dict)
                and payload.get("version") == DERIVED_CACHE_VERSION
                and payload.get("key") == key
            ):
                xs_cached = payload.get("xs")
                ys_cached = payload.get("ys_nested")
                desc_cached = payload.get("description")
                if xs_cached is not None and ys_cached is not None and desc_cached is not None:
                    logging.info(f"Loaded cached derived series for {desc_cached}")
                    return xs_cached, ys_cached, desc_cached
        except Exception as e:
            logging.error(f"Error loading cached derived series for {key}: {e}")
            pass  # fall through to rebuild

    xs, ys_nested, desc = build_derived_series(params)

    try:
        payload = {
            "version": DERIVED_CACHE_VERSION,
            "key": key,
            "params": params_d,
            "xs": xs,
            "ys_nested": ys_nested,
            "description": desc,
        }
        with cache_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.error(f"Error saving cached derived series for {key}: {e}")
        pass  # ignore cache write failures

    return xs, ys_nested, desc


@dataclass
class DerivedDatasetParameters:
    working_dir: str
    x_seconds_start_dropped: int = 0
    # Target final total seconds after all adjustments. If None or <= 0, don't enforce a total.
    x_seconds_total: Optional[int] = None
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
    DerivedDatasetParameters(
        working_dir="0726Midnight_WorkingDir",
        x_seconds_total=15000,
        vertical_scaling_factor=1.17,
        offset_compression_factor=0.5,
    ),
    DerivedDatasetParameters(
        working_dir="0620Midnight_WorkingDir",
        x_seconds_total=15000,
        vertical_scaling_factor=0.95,
        offset_compression_factor=0.8,
    ),
    DerivedDatasetParameters(
        working_dir="0621Midnight_WorkingDir",
        x_seconds_total=15000,
        vertical_scaling_factor=1.2,
        horizontal_scaling_factor=0.6,
        offset_compression_factor=0.7,
        augment_data=True,
    ),
    # Liner 1 100p, -47 @ 5500
    DerivedDatasetParameters(
        working_dir="0601_WorkingDir",
        x_seconds_total=15000,
        offset_compression_factor=0.85,
    ),
    DerivedDatasetParameters(
        working_dir="0614_WorkingDir",
        x_seconds_total=15000,
        offset_compression_factor=0.85,
    ),
    DerivedDatasetParameters(
        working_dir="0620_WorkingDir",
        x_seconds_total=15000,
        offset_compression_factor=0.95,
    ),
    # Liner 1 120p, -55 @ 6000

    # Liner 2 80p, -23 @ 3000
    # Liner 2 100p, -26 @ 3500
    # Liner 2 120p, -30 @ 4000
]


def _start_drop_index(*, num_entries: int, seconds_start_drop: int, interval_s: int) -> int:
    """Compute start index after dropping the requested head seconds (ceil to full frames)."""
    start_idx = int(np.ceil(seconds_start_drop / float(interval_s))) if seconds_start_drop > 0 else 0
    return min(start_idx, num_entries)


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


def _dropout_by_blocks(values: np.ndarray, keep_fraction: float, block_size: int = 100) -> np.ndarray:
    """Deterministically drop points in fixed-size blocks to keep ~keep_fraction of points.

    Points to keep within each block are sampled as evenly as possible to avoid clustering.
    """
    if keep_fraction >= 1.0:
        return values.copy()
    if keep_fraction <= 0.0:
        # Keep at least one point overall to avoid empty series
        return values[:1].copy()

    kept_indices: list[int] = []
    n = len(values)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_len = end - start
        keep_count = max(1, int(round(keep_fraction * block_len)))
        # Evenly spaced indices in [0, block_len-1]
        local = np.linspace(0, block_len - 1, num=keep_count, dtype=int)
        kept_indices.extend((start + local).tolist())

    kept_indices = sorted(set(kept_indices))
    return values[kept_indices]


def _fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y ≈ c + b * exp(-t / tau) by linearizing with an estimated asymptote c.

    Returns (c, b, tau). Falls back to a near-linear fit if degeneracies occur.
    """
    if len(x) < 2 or float(np.std(x)) == 0.0:
        c = float(y[-1] if len(y) > 0 else 0.0)
        b = float((y[0] - c) if len(y) > 0 else 0.0)
        tau = float(x[-1] - x[0] + 1.0) if len(x) > 1 else 1.0
        return c, b, tau

    # Estimate asymptote as the mean of the last 10% (min 5) points
    tail = max(5, int(0.1 * len(y)))
    c = float(np.mean(y[-tail:]))
    z = y - c
    # Ensure positivity for log, clamp very small values
    eps = 1e-12
    z = np.maximum(z, eps)
    lnz = np.log(z)

    # Linear regression: lnz ≈ ln(b) - t/τ
    slope, intercept = np.polyfit(x, lnz, 1)
    if slope >= 0:  # Non-decaying fit; fallback to small decay
        tau = max(1.0, (x[-1] - x[0]) / 3.0)
    else:
        tau = -1.0 / slope
    b = float(np.exp(intercept))
    return c, b, tau


def _extend_with_exponential(values: np.ndarray, interval_s: int, factor: float, noise_std: float) -> np.ndarray:
    """Extend the series length by factor (>1) using exponential decay fit plus Gaussian noise."""
    n = len(values)
    if n == 0 or factor <= 1.0:
        return values.copy()

    target_n = int(round(factor * n))
    extra = max(0, target_n - n)
    if extra == 0:
        return values.copy()

    x = np.arange(n, dtype=float) * float(interval_s)
    c, b, tau = _fit_exponential_decay(x, values)
    x_extra = (np.arange(n, n + extra, dtype=float)) * float(interval_s)
    y_extra_mean = c + b * np.exp(-(x_extra - x[0]) / tau)

    rng = np.random.default_rng()
    y_extra = y_extra_mean + rng.normal(0.0, noise_std, size=extra)
    return np.concatenate([values, y_extra.astype(float)])


def _extend_exponential_to_length(values: np.ndarray, interval_s: int, target_n: int, noise_std: float) -> np.ndarray:
    """Extend the series to exactly target_n points using an exponential decay fit plus noise."""
    n = len(values)
    if target_n <= n:
        return values.copy()

    x = np.arange(n, dtype=float) * float(interval_s)
    c, b, tau = _fit_exponential_decay(x, values)
    extra = target_n - n
    x_extra = (np.arange(n, n + extra, dtype=float)) * float(interval_s)
    y_extra_mean = c + b * np.exp(-(x_extra - x[0]) / tau)
    rng = np.random.default_rng()
    y_extra = y_extra_mean + rng.normal(0.0, noise_std, size=extra)
    return np.concatenate([values, y_extra.astype(float)])


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
    start_idx = _start_drop_index(
        num_entries=len(ys_nested),
        seconds_start_drop=params.x_seconds_start_dropped,
        interval_s=interval,
    )

    # Apply head drop only; no tail trimming here
    xs = xs[start_idx:]
    ys_nested = ys_nested[start_idx:]

    # Apply vertical (measurement) scaling if requested
    if params.vertical_scaling_factor != 1.0:
        ys_nested = [[y * params.vertical_scaling_factor for y in sub] for sub in ys_nested]

    # Compute the target total number of samples (always set)
    desired_n_total = (
        int(np.ceil(params.x_seconds_total / float(interval)))
        if (params.x_seconds_total is not None and params.x_seconds_total > 0)
        else len(ys_nested)
    )

    # Compute per-image means and apply noise suppression first
    means = np.array([float(np.mean(sub)) for sub in ys_nested], dtype=float)
    means_suppressed = _apply_noise_suppression(
        means,
        window=params.moving_average_window_points,
        compress=params.offset_compression_factor,
    )

    # Determine noise level from the last window of suppressed residuals against an exponential fit
    # (used for exponential extension noise below)
    x_arr = np.array(xs, dtype=float)
    if len(x_arr) >= 2 and np.std(x_arr) > 0:
        c_fit, b_fit, tau_fit = _fit_exponential_decay(x_arr, means_suppressed)
        fitted_exp = c_fit + b_fit * np.exp(-(x_arr - x_arr[0]) / tau_fit)
        residuals_suppressed = means_suppressed - fitted_exp
    else:
        residuals_suppressed = np.zeros_like(means_suppressed)
    win_pts = params.moving_average_window_points if params.moving_average_window_points > 1 else len(residuals_suppressed)
    win_pts = min(win_pts, len(residuals_suppressed)) if len(residuals_suppressed) > 0 else 0
    base_noise_std = float(np.std(residuals_suppressed[-win_pts:])) if win_pts > 0 else 0.0

    # Apply new horizontal scaling semantics
    series_after_h: np.ndarray
    if params.horizontal_scaling_factor < 1.0:
        keep_fraction = max(0.0, min(1.0, params.horizontal_scaling_factor))
        series_after_h = _dropout_by_blocks(means_suppressed, keep_fraction, block_size=100)
    elif params.horizontal_scaling_factor > 1.0:
        series_after_h = _extend_with_exponential(means_suppressed, interval_s=interval, factor=params.horizontal_scaling_factor, noise_std=base_noise_std * params.augmentation_noise_scale)
    else:
        series_after_h = means_suppressed

    # Rebuild xs as uniform grid with original interval after horizontal scaling
    xs = [i * interval for i in range(len(series_after_h))]

    # After horizontal scaling, if still shorter than target total seconds, extend to reach it
    # Gate this extension on explicit opt-in via augment_data
    if len(series_after_h) < desired_n_total and params.augment_data:
        series_after_h = _extend_exponential_to_length(
            series_after_h,
            interval_s=interval,
            target_n=desired_n_total,
            noise_std=base_noise_std * params.augmentation_noise_scale,
        )
        xs = [i * interval for i in range(len(series_after_h))]
    
    # If longer than target total, trim to the desired length (always applied)
    if len(series_after_h) > desired_n_total:
        series_after_h = series_after_h[:desired_n_total]
        xs = [i * interval for i in range(len(series_after_h))]

    ys_nested_adjusted = [[float(v)] for v in series_after_h.tolist()]

    return xs, ys_nested_adjusted, params.description


def export_derived_y_plots(derived_params: List[DerivedDatasetParameters]) -> None:
    if len(derived_params) == 0:
        logging.info("No derived dataset parameters specified; nothing to export.")
        return

    # Sort using label override when present
    # derived_params_sorted = sorted(derived_params, key=lambda p: p.effective_label.value)

    output_file_y = PROJECT_ROOT / "derived_analysis_output_displacement_y.pdf"

    plots = []
    for dp in derived_params:
        xs, nested_ys, desc = load_or_build_derived_series(dp)
        fig = plot_dataset(xs, nested_ys, desc, "derived displacement y")
        plots.append(fig)

    with PdfPages(output_file_y) as pdf:
        for fig in plots:
            pdf.savefig(fig)
            plt.close(fig)

    logging.info(f"Exported derived Y displacement plots to: {output_file_y}")


if __name__ == "__main__":
    setup_logging()
    export_derived_y_plots(DERIVED_DATASET_PARAMETERS)

