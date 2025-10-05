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
    snippet_lines.append(r"\begin{table}[htbp]")
    snippet_lines.append(r"\centering")
    snippet_lines.append(r"\begin{tabular}{lrrrr}")
    snippet_lines.append(r"\toprule")
    snippet_lines.append(" {} \\\\".format(" & ".join(header_cols)))
    snippet_lines.append(r"\midrule")

    def _fmt(v: float) -> str:
        return f"{v:.2f}"

    for label in labels:
        row_vals: List[str] = []
        row_vals.append(_latex_escape(label.pretty_string))
        results = labels_to_results[label]
        means = [r.y_mean_last_seconds for r in results]
        row_vals.extend([_fmt(v) for v in means])
        avg_val = float(np.mean(means)) if len(means) > 0 else float("nan")
        row_vals.append(_fmt(avg_val))
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
    def _avg_for_label(label: ExperimentLabels) -> float:
        if label not in labels_to_results:
            raise ValueError(f"Missing results for label {label}")
        results = labels_to_results[label]
        if len(results) != 3:
            raise AssertionError(
                f"Label {label} has {len(results)} results; expected exactly 3"
            )
        vals = [r.y_mean_last_seconds for r in results]
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    def _fmt(v: float) -> str:
        return f"{v:.2f}"

    # Build LaTeX table snippet with header row and left label column
    snippet_lines: List[str] = []
    snippet_lines.append(r"\begin{table}[htbp]")
    snippet_lines.append(r"\centering")
    snippet_lines.append(r"\begin{tabular}{lrrr}")
    snippet_lines.append(r"\toprule")

    # Header row for columns
    header_cols = [
        "",
        "80 percent",
        "100 percent",
        "120 percent",
    ]
    snippet_lines.append(r" {} \\".format(" & ".join(header_cols)))
    snippet_lines.append(r"\midrule")

    # Two rows: Liner 1 then Liner 2, with left label column
    row_titles = ["Liner 1", "Liner 2"]
    for row_idx, row_labels in enumerate(row_definitions):
        row_vals: List[str] = []
        row_vals.append(_latex_escape(row_titles[row_idx]))
        for label in row_labels:
            row_vals.append(_fmt(_avg_for_label(label)))
        snippet_lines.append(" {} \\\\".format(" & ".join(row_vals)))

    snippet_lines.append(r"\bottomrule")
    snippet_lines.append(r"\end{tabular}")
    label = f"Trials Average of Mean Displacement Y ({MICROMETER}) in last {RESULT_PARAMETERS.last_x_minutes} minutes"
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
    export_six_by_three_figure(labels_to_trials)
    export_results_table_figure(labels_to_results, [
        ExperimentLabels.LINER_1_80P,
        ExperimentLabels.LINER_1_100P,
        ExperimentLabels.LINER_1_120P,
    ], "liner_1_results_table", f"Liner 1 Mean Displacement Y ({MICROMETER}) in last {RESULT_PARAMETERS.last_x_minutes} minutes")
    export_results_table_figure(labels_to_results, [
        ExperimentLabels.LINER_2_80P,
        ExperimentLabels.LINER_2_100P,
        ExperimentLabels.LINER_2_120P,
    ], "liner_2_results_table", f"Liner 2 Mean Displacement Y ({MICROMETER}) in last {RESULT_PARAMETERS.last_x_minutes} minutes")
    export_avg_results_table_figure()
    clean_unused_derived_cache()


if __name__ == "__main__":
    export_all_figures()
