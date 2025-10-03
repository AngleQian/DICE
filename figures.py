from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
from dataclasses import dataclass

from parameters import ExperimentLabels
from analysis import PROJECT_ROOT, MICROMETER
from derived_datasets import (
    DERIVED_DATASET_PARAMETERS,
    DerivedDatasetParameters,
    clean_unused_derived_cache,
    load_or_build_derived_series,
)
from logging_utils import setup_logging
from result import DatasetResult, get_dataset_result


FINAL_RESULT_WINDOW_TIME_SECONDS = 30 * 60


def _set_academic_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Use Latin Modern for better T1 output (very close to Computer Modern)
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    })


def _collect_trials_by_label(
    derived_params: List[DerivedDatasetParameters],
) -> Dict[ExperimentLabels, List[Tuple[List[float], List[List[float]], str]]]:
    """Build mapping label -> list of (xs, nested_ys, description) for available trials."""
    by_label: Dict[ExperimentLabels, List[Tuple[List[float], List[List[float]], str]]] = {}
    for dp in derived_params:
        label = dp.effective_label
        xs, nested_ys, desc = load_or_build_derived_series(dp)
        by_label.setdefault(label, []).append((xs, nested_ys, desc))
    return by_label


def _collect_results_by_label(
    derived_params: List[DerivedDatasetParameters],
) -> Dict[ExperimentLabels, DatasetResult]:
    """Build mapping label -> DatasetResult for available derived datasets."""
    by_label: Dict[ExperimentLabels, DatasetResult] = {}
    for dp in derived_params:
        label = dp.effective_label
        result = get_dataset_result(dp)
        by_label[label] = result
    return by_label


def export_six_by_three_figure(
    labels_to_trials: Dict[ExperimentLabels, List[Tuple[List[float], List[List[float]], str]]]
) -> None:
    """Export a 6x3 grid figure: rows are labels, columns are trials per label."""
    logging.info(f"Exporting 6x3 figure")
    _set_academic_style()

    row_labels: List[ExperimentLabels] = [
        ExperimentLabels.LINER_1_80P,
        ExperimentLabels.LINER_1_100P,
        ExperimentLabels.LINER_1_120P,
        ExperimentLabels.LINER_2_80P,
        ExperimentLabels.LINER_2_100P,
        ExperimentLabels.LINER_2_120P,
    ]

    n_rows, n_cols = 6, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=True, sharey=True)

    # Column titles
    for col in range(n_cols):
        axes[0, col].set_title(f"Trial {col + 1}")

    # Plot each row
    for r, label in enumerate(row_labels):
        if label not in labels_to_trials:
            raise ValueError(f"Label {label} not found in labels_to_trials")
        trials = labels_to_trials[label]
        if len(trials) != n_cols:
            raise ValueError(f"Label {label} has {len(trials)} trials, expected {n_cols}")
        for c in range(n_cols):
            ax = axes[r, c]
            xs, nested_ys, desc = trials[c]
            ys_means = np.array([np.mean(y) for y in nested_ys], dtype=float)
            ax.plot(xs, ys_means, "-", color="black", linewidth=0.9)

            # Row label on the left-most column
            if c == 0:
                ax.set_ylabel(label.pretty_string)

            # Only bottom row gets x label ticks label
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")

    # Common Y label on entire figure (left side)
    fig.text(0.005, 0.5, f"Displacement y ({MICROMETER})", va="center", rotation="vertical")

    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.98))

    output_path = PROJECT_ROOT / "six_by_three_figure.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_all_figures() -> None:
    """Main entrypoint to export all figures in one shot."""
    setup_logging()
    labels_to_trials = _collect_trials_by_label(DERIVED_DATASET_PARAMETERS)
    labels_to_results = _collect_results_by_label(DERIVED_DATASET_PARAMETERS)
    export_six_by_three_figure(labels_to_trials)
    clean_unused_derived_cache()


if __name__ == "__main__":
    export_all_figures()

