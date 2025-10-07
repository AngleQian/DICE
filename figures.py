from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
from dataclasses import dataclass
import subprocess

from parameters import ExperimentLabels
from analysis import PROJECT_ROOT, MICROMETER
from derived_datasets import (
    DERIVED_DATASET_PARAMETERS,
    DerivedDatasetParameters,
    clean_unused_derived_cache,
    load_or_build_derived_series,
)
from logging_utils import setup_logging
from result import RESULT_PARAMETERS, DatasetResult, get_dataset_result


FIGURE_DIR = PROJECT_ROOT / "figures_artifacts"


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
) -> Dict[ExperimentLabels, List[DatasetResult]]:
    """Build mapping label -> DatasetResult for available derived datasets."""
    by_label: Dict[ExperimentLabels, DatasetResult] = {}
    for dp in derived_params:
        label = dp.effective_label
        result = get_dataset_result(dp)
        by_label.setdefault(label, []).append(result)
    return by_label


def export_six_by_three_figure(
    labels_to_trials: Dict[ExperimentLabels, List[Tuple[List[float], List[List[float]], str]]],
    labels_to_results: Dict[ExperimentLabels, List[DatasetResult]],
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
        if label not in labels_to_results:
            raise ValueError(f"Label {label} not found in labels_to_results")
        results_for_label = labels_to_results[label]
        if len(results_for_label) != n_cols:
            raise ValueError(f"Label {label} has {len(results_for_label)} results, expected {n_cols}")
        for c in range(n_cols):
            ax = axes[r, c]
            xs, nested_ys, desc = trials[c]
            ys_means = np.array([np.mean(y) for y in nested_ys], dtype=float)
            ax.plot(xs, ys_means, "-", color="black", linewidth=0.9)

            # Add vertical shaded region for the last-X-seconds window
            res = results_for_label[c]
            start = float(res.last_x_seconds_start)
            end = float(res.last_x_seconds_end)
            ax.axvspan(start, end, color="tab:grey", alpha=0.08)

            # Row label on the left-most column
            if c == 0:
                ax.set_ylabel(label.pretty_string)

            # Only bottom row gets x label ticks label
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")

    # Common Y label on entire figure (left side)
    fig.text(0.005, 0.5, f"Displacement y ({MICROMETER})", va="center", rotation="vertical")

    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.98))

    output_path = FIGURE_DIR / "six_by_three_figure.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def compile_table_snippet_preview(figure_name: str) -> None:
    """Create a standalone wrapper for `{figure_name}.tex` and compile `{figure_name}.pdf`.

    This lets us embed `{figure_name}.tex` in larger papers while still generating
    a preview PDF for quick inspection.
    """
    # Write standalone wrapper to compile PDF preview using the snippet
    wrapper_lines: List[str] = []
    wrapper_lines.append(r"\documentclass{article}")
    wrapper_lines.append(r"\usepackage{booktabs}")
    wrapper_lines.append(r"\usepackage[T1]{fontenc}")
    wrapper_lines.append(r"\usepackage{lmodern}")
    wrapper_lines.append(r"\begin{document}")
    wrapper_lines.append(f"\\input{{{figure_name}.tex}}")
    wrapper_lines.append(r"\end{document}")
    wrapper_src = "\n".join(wrapper_lines)

    wrapper_path = FIGURE_DIR / f"{figure_name}_wrapper.tex"
    with wrapper_path.open("w", encoding="utf-8") as fh:
        fh.write(wrapper_src)

    # Compile to PDF using pdflatex
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-jobname={figure_name}",
                f"{figure_name}_wrapper.tex",
            ],
            cwd=str(FIGURE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        logging.info(f"Compiled LaTeX to PDF: {(FIGURE_DIR / (figure_name + '.pdf'))}")
    except subprocess.CalledProcessError as e:
        log_path = FIGURE_DIR / f"{figure_name}.log"
        logging.error(
            f"pdflatex failed with code {e.returncode}. See log at {log_path if log_path.exists() else 'N/A'}"
        )
        raise


def export_results_table_figure(
    labels_to_results: Dict[ExperimentLabels, List[DatasetResult]],
    labels: List[ExperimentLabels],
    figure_name: str,
    figure_label: str,
) -> str:
    """
    Build a LaTeX table figure summarizing per-label results.

    - First column: label pretty string
    - Next 3 columns: each label's three DatasetResult values (y_mean_last_seconds)
    - Final column: average of the three values

    Writes {figure_name}.tex and compiles {figure_name}.pdf in FIGURE_DIR.

    Returns the LaTeX source string.
    """
    # Validate inputs
    if len(labels) == 0:
        raise ValueError("labels list is empty")

    for label in labels:
        if label not in labels_to_results:
            raise ValueError(f"Missing results for label {label}")
        results = labels_to_results[label]
        if len(results) != 3:
            raise AssertionError(
                f"Label {label} has {len(results)} results; expected exactly 3 for 5 total columns"
            )

    # Build table rows (embeddable snippet with table environment)
    header_cols = ["", "Trial 1", "Trial 2", "Trial 3", "Average"]
    snippet_lines: List[str] = []
    snippet_lines.append(r"\begin{table}[htbp!]")
    snippet_lines.append(r"\centering")
    snippet_lines.append(r"\begin{tabular}{lrrrr}")
    snippet_lines.append(r"\toprule")
    snippet_lines.append(" {} \\\\".format(" & ".join(header_cols)))
    snippet_lines.append(r"\midrule")

    def _fmt(v: float) -> str:
        return f"{v:.1f}"

    for label in labels:
        row_vals: List[str] = []
        row_vals.append(label.pretty_string)
        results = labels_to_results[label]
        means = [r.y_mean_last_seconds for r in results]
        means_absolute = [(r.y_mean_last_seconds, r.absolute_uncertainty) for r in results]
        standard_error = float(np.std(means, ddof=1)) / (len(means) ** 0.5)
        row_vals.extend([f"${_fmt(m)} \pm {_fmt(a)}$" for m, a in means_absolute])
        avg_val = float(np.mean(means)) if len(means) > 0 else float("nan")
        row_vals.append(f"${_fmt(avg_val)} \pm {_fmt(standard_error)}$")
        snippet_lines.append(" {} \\\\".format(" & ".join(row_vals)))

    snippet_lines.append(r"\bottomrule")
    snippet_lines.append(r"\end{tabular}")
    snippet_lines.append(f"\\caption{{{_latex_escape(figure_label)}}}")
    snippet_lines.append(f"\\label{{fig:{figure_name}}}")
    snippet_lines.append(r"\end{table}")

    latex_snippet_src = "\n".join(snippet_lines)
    # Write embeddable snippet: {figure_name}.tex
    snippet_path = FIGURE_DIR / f"{figure_name}.tex"
    with snippet_path.open("w", encoding="utf-8") as fh:
        fh.write(latex_snippet_src)

    # Compile preview PDF via reusable wrapper utility
    compile_table_snippet_preview(figure_name)
    return latex_snippet_src


def export_avg_results_plot_figure() -> str:
    """
    Produce a vertical bar plot of averages of `y_mean_last_seconds` for six configurations.

    Bars in order:
      Liner 1 80%, Liner 1 100%, Liner 1 120%, Liner 2 80%, Liner 2 100%, Liner 2 120%

    Error bars are standard errors across the three trials per configuration.

    Writes avg_results_plot.pdf in FIGURE_DIR.

    Returns the output path string.
    """
    labels_to_results = _collect_results_by_label(DERIVED_DATASET_PARAMETERS)

    label_order: List[ExperimentLabels] = [
        ExperimentLabels.LINER_1_80P,
        ExperimentLabels.LINER_1_100P,
        ExperimentLabels.LINER_1_120P,
        ExperimentLabels.LINER_2_80P,
        ExperimentLabels.LINER_2_100P,
        ExperimentLabels.LINER_2_120P,
    ]

    # Validate presence and compute means and standard errors
    means: List[float] = []
    standard_errors: List[float] = []
    xticklabels: List[str] = []
    for label in label_order:
        if label not in labels_to_results:
            raise ValueError(f"Missing results for label {label}")
        results = labels_to_results[label]
        if len(results) != 3:
            raise AssertionError(
                f"Label {label} has {len(results)} results; expected exactly 3"
            )
        vals = [float(r.y_mean_last_seconds) for r in results]
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / (len(vals) ** 0.5)
        means.append(mean)
        standard_errors.append(se)
        xticklabels.append(label.pretty_string)

    _set_academic_style()

    fig: Figure
    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(label_order))
    barlist = ax.bar(x, means, yerr=standard_errors, capsize=3, color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_ylabel(f"Displacement y ({MICROMETER})")
    ax.set_title("Average steady-state Y displacements")

    fig.tight_layout()

    output_path = FIGURE_DIR / "avg_results_plot.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def export_results_plot_figure(
    labels_to_results: Dict[ExperimentLabels, List[DatasetResult]],
    labels: List[ExperimentLabels],
    figure_name: str,
    figure_label: str,
) -> str:
    """
    Build a vertical bar plot summarizing per-label results.

    For each label, draws 4 bars in order:
      Trial 1, Trial 2, Trial 3, Average

    Error bars:
      - Trials use asymmetric errors from p05 and p95 relative to the mean
      - Average uses standard error across the three trials (same as table)

    Saves {figure_name}.pdf in FIGURE_DIR and returns the output path string.
    """
    # Validate inputs
    if len(labels) == 0:
        raise ValueError("labels list is empty")

    for label in labels:
        if label not in labels_to_results:
            raise ValueError(f"Missing results for label {label}")
        results = labels_to_results[label]
        if len(results) != 3:
            raise AssertionError(
                f"Label {label} has {len(results)} results; expected exactly 3"
            )

    _set_academic_style()

    # Prepare bar positions and values
    x_positions: List[float] = []
    bar_heights: List[float] = []
    # Asymmetric yerr for trials (lower, upper). For average we use symmetric SE
    lower_errors: List[float] = []
    upper_errors: List[float] = []
    is_average_bar: List[bool] = []

    # Track group centers for x tick labels
    group_centers: List[float] = []
    group_labels: List[str] = []

    pos_cursor = 0.0
    group_gap = 0.6  # visual gap between label groups

    for label in labels:
        results = labels_to_results[label]
        means = [float(r.y_mean_last_seconds) for r in results]
        p05s = [float(r.y_p05_last_seconds) for r in results]
        p95s = [float(r.y_p95_last_seconds) for r in results]

        group_start = pos_cursor
        # Trials 1..3
        for i in range(3):
            x_positions.append(pos_cursor)
            bar_heights.append(means[i])
            lower_errors.append(max(0.0, means[i] - p05s[i]))
            upper_errors.append(max(0.0, p95s[i] - means[i]))
            is_average_bar.append(False)
            pos_cursor += 1.0

        # Average bar with standard error
        avg_val = float(np.mean(means)) if len(means) > 0 else float("nan")
        standard_error = float(np.std(means, ddof=1)) / (len(means) ** 0.5)
        x_positions.append(pos_cursor)
        bar_heights.append(avg_val)
        lower_errors.append(standard_error)
        upper_errors.append(standard_error)
        is_average_bar.append(True)

        # Compute center of this group for x tick label
        group_end = pos_cursor
        group_centers.append((group_start + group_end) / 2.0)
        group_labels.append(label.pretty_string)

        # Advance cursor to next group start
        pos_cursor += 1.0 + group_gap  # move past the avg bar and add gap

    fig: Figure
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Choose colors: trials vs average
    trial_color = "tab:blue"
    avg_color = "tab:orange"
    colors = [avg_color if flag else trial_color for flag in is_average_bar]

    # Build asymmetric yerr: shape (2, N)
    yerr = np.vstack([np.array(lower_errors), np.array(upper_errors)])

    ax.bar(
        x_positions,
        bar_heights,
        color=colors,
        width=0.8,
        yerr=yerr,
        capsize=3,
        linewidth=0.5,
        edgecolor="black",
    )

    # X axis labeling: group labels under centers
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=0)

    # Light vertical separators between groups to aid readability
    for center in group_centers:
        # place separator halfway between groups by using group_gap; approximate
        ax.axvline(center + 2.0 + (group_gap / 2.0), color="grey", alpha=0.15, linewidth=0.8)

    ax.set_ylabel(f"Displacement y ({MICROMETER})")
    ax.set_title(figure_label)

    # Create a small legend proxy
    trial_proxy = plt.Rectangle((0, 0), 1, 1, color=trial_color, ec="black", lw=0.5)
    avg_proxy = plt.Rectangle((0, 0), 1, 1, color=avg_color, ec="black", lw=0.5)
    ax.legend([trial_proxy, avg_proxy], ["Trial", "Average"], frameon=False, loc="best")

    fig.tight_layout()

    output_path = FIGURE_DIR / f"{figure_name}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def export_avg_results_table_figure() -> str:
    """
    Produce a 2x3 LaTeX table of averages of `y_mean_last_seconds`.

    Rows: Liner 1, Liner 2 (shown in a leftmost label column)
    Columns: 80 percent, 100 percent, 120 percent (shown in a header row)

    Each cell is the average of the three trials for that label.

    Writes avg_results_table.tex and compiles avg_results_table.pdf in FIGURE_DIR.

    Returns the LaTeX source string.
    """
    labels_to_results = _collect_results_by_label(DERIVED_DATASET_PARAMETERS)

    # Define grid ordering
    row_definitions: List[List[ExperimentLabels]] = [
        [
            ExperimentLabels.LINER_1_80P,
            ExperimentLabels.LINER_1_100P,
            ExperimentLabels.LINER_1_120P,
        ],
        [
            ExperimentLabels.LINER_2_80P,
            ExperimentLabels.LINER_2_100P,
            ExperimentLabels.LINER_2_120P,
        ],
    ]

    # Validate presence and compute per-label averages
    def _avg_std_error_for_label(label: ExperimentLabels) -> Tuple[float, float]:
        if label not in labels_to_results:
            raise ValueError(f"Missing results for label {label}")
        results = labels_to_results[label]
        if len(results) != 3:
            raise AssertionError(
                f"Label {label} has {len(results)} results; expected exactly 3"
            )
        vals = [r.y_mean_last_seconds for r in results]
        mean = float(np.mean(vals))
        standard_error = float(np.std(vals, ddof=1)) / (len(vals) ** 0.5)
        return mean, standard_error

    def _fmt(v: float) -> str:
        return f"{v:.1f}"

    # Build LaTeX table snippet with header row and left label column
    snippet_lines: List[str] = []
    snippet_lines.append(r"\begin{table}[htbp!]")
    snippet_lines.append(r"\centering")
    snippet_lines.append(r"\begin{tabular}{lrrr}")
    snippet_lines.append(r"\toprule")

    # Header row for columns
    header_cols = [
        "",
        "$80\%$",
        "$100\%$",
        "$120\%$",
    ]
    header_cols = [col for col in header_cols]
    snippet_lines.append(r" {} \\".format(" & ".join(header_cols)))
    snippet_lines.append(r"\midrule")

    # Two rows: Liner 1 then Liner 2, with left label column
    row_titles = ["Liner 1", "Liner 2"]
    for row_idx, row_labels in enumerate(row_definitions):
        row_vals: List[str] = []
        row_vals.append(_latex_escape(row_titles[row_idx]))
        for label in row_labels:
            mean, standard_error = _avg_std_error_for_label(label)
            row_vals.append(f"${_fmt(mean)} \pm {_fmt(standard_error)}$")
        snippet_lines.append(" {} \\\\".format(" & ".join(row_vals)))

    snippet_lines.append(r"\bottomrule")
    snippet_lines.append(r"\end{tabular}")
    label = f"Average steady-state Y displacements ({MICROMETER}) for all configurations"
    snippet_lines.append(f"\\caption{{{_latex_escape(label)}}}")
    snippet_lines.append(f"\\label{{fig:avg_results_table}}")
    snippet_lines.append(r"\end{table}")

    latex_snippet_src = "\n".join(snippet_lines)

    # Write embeddable snippet and compile PDF preview
    figure_name = "avg_results_table"
    snippet_path = FIGURE_DIR / f"{figure_name}.tex"
    with snippet_path.open("w", encoding="utf-8") as fh:
        fh.write(latex_snippet_src)

    compile_table_snippet_preview(figure_name)
    return latex_snippet_src

def export_all_figures() -> None:
    """Main entrypoint to export all figures in one shot."""
    setup_logging()

    labels_to_trials = _collect_trials_by_label(DERIVED_DATASET_PARAMETERS)
    labels_to_results = _collect_results_by_label(DERIVED_DATASET_PARAMETERS)

    export_six_by_three_figure(labels_to_trials, labels_to_results)
    export_results_table_figure(labels_to_results, [
        ExperimentLabels.LINER_1_80P,
        ExperimentLabels.LINER_1_100P,
        ExperimentLabels.LINER_1_120P,
    ], "liner_1_results_table", f"Steady-state Y displacements ({MICROMETER}) for Liner 1")
    export_results_table_figure(labels_to_results, [
        ExperimentLabels.LINER_2_80P,
        ExperimentLabels.LINER_2_100P,
        ExperimentLabels.LINER_2_120P,
    ], "liner_2_results_table", f"Steady-state Y displacements ({MICROMETER}) for Liner 2")
    export_results_plot_figure(labels_to_results, [
        ExperimentLabels.LINER_1_80P,
        ExperimentLabels.LINER_1_100P,
        ExperimentLabels.LINER_1_120P,
    ], "liner_1_results_plot", f"Steady-state Y displacements ({MICROMETER}) for Liner 1")
    export_results_plot_figure(labels_to_results, [
        ExperimentLabels.LINER_2_80P,
        ExperimentLabels.LINER_2_100P,
        ExperimentLabels.LINER_2_120P,
    ], "liner_2_results_plot", f"Steady-state Y displacements ({MICROMETER}) for Liner 2")
    export_avg_results_table_figure()
    export_avg_results_plot_figure()
    clean_unused_derived_cache()


if __name__ == "__main__":
    export_all_figures()
