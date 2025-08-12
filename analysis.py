import csv
from functools import cached_property
import pathlib
from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable
from parameters import (
    ExperimentLabels,
    DatasetParameters,
    AnalysisParameters,
    ANALYSIS_PARAMETERS,
    find_dataset_parameters,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import logging
import sys
from multiprocessing import get_context


PROJECT_ROOT = pathlib.Path("/Users/xuxin/Documents/Dice/")
MICROMETER = "\u00b5m"


def get_results_path(working_dir: str) -> pathlib.Path:
    return PROJECT_ROOT / working_dir / "results"


def get_solution_files(working_dir: str) -> list[pathlib.Path]:
    results_path = get_results_path(working_dir)
    result_files = list(results_path.glob("DICe_solution_*.txt"))
    return sorted(result_files, key=lambda x: x.name)


@dataclass
class SolutionFileRow:
    # Raw data in DICE units (px)
    subset_id: int
    coordinate_x: float
    coordinate_y: float
    displacement_x: float
    displacement_y: float
    sigma: float
    gamma: float
    beta: float
    status_flag: int
    uncertainty: float
    vsg_strain_xx: float
    vsg_strain_yy: float
    vsg_strain_xy: float

    @property
    def total_displacement(self) -> float:
        return (self.displacement_x**2 + self.displacement_y**2) ** 0.5

    @classmethod
    def from_csv_row(cls, row: list[str], header: list[str]) -> "SolutionFileRow":
        return cls(
            subset_id=cls._extract_int_value(row, header, 0, "SUBSET_ID"),
            coordinate_x=cls._extract_float_value(row, header, 1, "COORDINATE_X"),
            coordinate_y=cls._extract_float_value(row, header, 2, "COORDINATE_Y"),
            displacement_x=cls._extract_float_value(row, header, 3, "DISPLACEMENT_X"),
            displacement_y=cls._extract_float_value(row, header, 4, "DISPLACEMENT_Y"),
            sigma=cls._extract_float_value(row, header, 5, "SIGMA"),
            gamma=cls._extract_float_value(row, header, 6, "GAMMA"),
            beta=cls._extract_float_value(row, header, 7, "BETA"),
            status_flag=cls._extract_float_value(row, header, 8, "STATUS_FLAG"),
            uncertainty=cls._extract_float_value(row, header, 9, "UNCERTAINTY"),
            vsg_strain_xx=cls._extract_float_value(row, header, 10, "VSG_STRAIN_XX"),
            vsg_strain_yy=cls._extract_float_value(row, header, 11, "VSG_STRAIN_YY"),
            vsg_strain_xy=cls._extract_float_value(row, header, 12, "VSG_STRAIN_XY"),
        )

    @classmethod
    def _extract_row_value(
        cls, row: list[str], header: list[str], idx: int, expected_header: str
    ) -> str:
        if header[idx] != expected_header:
            raise ValueError(
                f'Expected header "{expected_header}" but got "{header[idx]}" at index {idx}'
            )
        return row[idx]

    @classmethod
    def _extract_int_value(
        cls, row: list[str], header: list[str], idx: int, expected_header: str
    ) -> int:
        value = cls._extract_float_value(row, header, idx, expected_header)
        assert (
            value.is_integer()
        ), f"Expected integer value for {expected_header}, but got {value}"
        return int(value)

    @classmethod
    def _extract_float_value(
        cls, row: list[str], header: list[str], idx: int, expected_header: str
    ) -> float:
        value = cls._extract_row_value(row, header, idx, expected_header)
        return float(value)


@dataclass
class LengthAdjustmentParams:
    px_to_m: float
    px_value_as_zero: float
    scale_to_us: bool = True
    angle_adjustment: float = 1.0 # Multiplier to adjust for angle between measurement direction and actual direction


@dataclass
class DataEntry:
    # e.g /Users/xuxin/Documents/Dice/0620_WorkingDir/results/DICe_solution_7946.txt
    solution_file: pathlib.Path
    subset_id_to_row: dict[int, SolutionFileRow]

    def __post_init__(self):
        assert (
            len(self.subset_id_to_row) > 0
        ), "Data entry must contain at least one subset ID"

    @property
    def image_index(self) -> int:
        filename_without_extension = self.solution_file.stem
        return int(filename_without_extension.split("_")[-1])

    @property
    def all_displacement_ys_px_raw(self) -> list[float]:
        return [row.displacement_y for row in self.subset_id_to_row.values()]

    @property
    def all_displacement_xs_px_raw(self) -> list[float]:
        return [row.displacement_x for row in self.subset_id_to_row.values()]

    @property
    def all_total_displacements_px_raw(self) -> list[float]:
        return [row.total_displacement for row in self.subset_id_to_row.values()]

    @cached_property
    def mean_displacement_y_px(self) -> float:
        return np.mean(self.all_displacement_ys_px_raw)

    @cached_property
    def mean_displacement_x_px(self) -> float:
        return np.mean(self.all_displacement_xs_px_raw)

    @cached_property
    def mean_total_displacement_px(self) -> float:
        return np.mean(self.all_total_displacements_px_raw)

    def get_all_displacement_ys_adjusted(
        self,
        length_adjustment_params: LengthAdjustmentParams,
    ) -> list[float]:
        return self._get_lengths_px_to_adjusted(
            self.all_displacement_ys_px_raw,
            length_adjustment_params,
        )

    def get_all_displacement_xs_adjusted(
        self,
        length_adjustment_params: LengthAdjustmentParams,
    ) -> list[float]:
        return self._get_lengths_px_to_adjusted(
            self.all_displacement_xs_px_raw,
            length_adjustment_params,
        )

    def get_all_total_displacements_adjusted(
        self,
        length_adjustment_params: LengthAdjustmentParams,
    ) -> list[float]:
        return self._get_lengths_px_to_adjusted(
            self.all_total_displacements_px_raw,
            length_adjustment_params,
        )

    @staticmethod
    def _get_lengths_px_to_adjusted(
        lengths_px: list[float],
        length_adjustment_params: LengthAdjustmentParams,
    ) -> list[float]:
        scale_factor = 1.0
        if length_adjustment_params.scale_to_us:
            scale_factor = 1e6
        return [
            (px - length_adjustment_params.px_value_as_zero)
            * length_adjustment_params.px_to_m
            * scale_factor
            for px in lengths_px
        ]


@dataclass
class Dataset:
    parameters: DatasetParameters
    data_entries: list[DataEntry]

    def __post_init__(self):
        assert (
            len(self.data_entries) > 0
        ), "Dataset must contain at least one data entry"
        # Check data entiries are sorted ascending image index
        prev_data_entry: Optional[DataEntry] = None
        for data_entry in self.data_entries:
            if (
                prev_data_entry is not None
                and data_entry.image_index < prev_data_entry.image_index
            ):
                raise ValueError(
                    f"Data entries are not sorted by image index: {prev_data_entry.image_index=} < {data_entry.image_index=}. {data_entry.solution_file=} < {prev_data_entry.solution_file=}"
                )
            prev_data_entry = data_entry

    @property
    def solution_directory(self) -> pathlib.Path:
        return get_results_path(self.working_dir)

    @property
    def min_image_index(self) -> int:
        return self.data_entries[0].image_index

    @property
    def max_image_index(self) -> int:
        return self.data_entries[-1].image_index

    @property
    def data_entry_xs_s(self) -> list[int]:
        return list(
            range(
                0,
                len(self.data_entries) * self.parameters.time_interval_per_image_s,
                self.parameters.time_interval_per_image_s,
            )
        )

    def get_displacement_ys_adjusted(
        self, *, start_index: int = 0, end_index: Optional[int] = None
    ) -> list[list[float]]:
        return self._get_length_measurement_adjusted(
            data_entry_to_mean_measurement_fn=lambda data_entry: data_entry.mean_displacement_y_px,
            data_entry_to_adjusted_measurement_fn=lambda data_entry, params: data_entry.get_all_displacement_ys_adjusted(
                params
            ),
            angle_adjustment=self.parameters.frame_y_to_bolt_y_adjustment,
            start_index=start_index,
            end_index=end_index,
        )

    def get_displacement_xs_adjusted(
        self, *, start_index: int = 0, end_index: Optional[int] = None
    ) -> list[list[float]]:
        return self._get_length_measurement_adjusted(
            data_entry_to_mean_measurement_fn=lambda data_entry: data_entry.mean_displacement_x_px,
            data_entry_to_adjusted_measurement_fn=lambda data_entry, params: data_entry.get_all_displacement_xs_adjusted(
                params
            ),
            angle_adjustment=self.parameters.frame_y_to_bolt_y_adjustment,
            start_index=start_index,
            end_index=end_index,
        )

    def get_total_displacements_adjusted(
        self, *, start_index: int = 0, end_index: Optional[int] = None
    ) -> list[list[float]]:
        return self._get_length_measurement_adjusted(
            data_entry_to_mean_measurement_fn=lambda data_entry: data_entry.mean_total_displacement_px,
            data_entry_to_adjusted_measurement_fn=lambda data_entry, params: data_entry.get_all_total_displacements_adjusted(
                params
            ),
            angle_adjustment=1.0,  # Total displacement does not need angle adjustment
            start_index=start_index,
            end_index=end_index,
        )

    def _get_length_measurement_adjusted(
        self,
        *,
        data_entry_to_mean_measurement_fn: Callable[[DataEntry], float],
        data_entry_to_adjusted_measurement_fn: Callable[
            [DataEntry, LengthAdjustmentParams], list[float]
        ],
        angle_adjustment: float,
        start_index: int,
        end_index: Optional[int],
    ) -> list[list[float]]:
        if end_index is None:
            end_index = len(self.data_entries)
        if self.parameters.first_n_mean_as_zero == 0:
            px_value_as_zero = 0.0
        else:
            px_value_as_zero = np.mean(
                [
                    data_entry_to_mean_measurement_fn(data_entry)
                    for data_entry in self.data_entries[
                        : self.parameters.first_n_mean_as_zero
                    ]
                ]
            )

        return [
            data_entry_to_adjusted_measurement_fn(
                data_entry,
                LengthAdjustmentParams(
                    px_to_m=self.parameters.px_to_m,
                    px_value_as_zero=px_value_as_zero,
                    scale_to_us=True,
                    angle_adjustment=angle_adjustment,
                ),
            )
            for data_entry in self.data_entries[start_index:end_index]
        ]


def load_dataentry(solution_file: pathlib.Path) -> DataEntry:
    """
    SUBSET_ID,COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,SIGMA,GAMMA,BETA,STATUS_FLAG,UNCERTAINTY,VSG_STRAIN_XX,VSG_STRAIN_YY,VSG_STRAIN_XY
    0,3.1400E+03,2.9600E+03,3.6102E-01,-1.0651E+00,5.0257E-03,2.1700E-02,0.0000E+00,4.0000E+00,7.6666E-03,-6.4999E-04,-4.3243E-04,5.0214E-04
    """
    with solution_file.open("r") as file:
        reader = csv.reader(file)
        header = next(reader)

        subset_id_to_row = {}
        for row in reader:
            solution_row = SolutionFileRow.from_csv_row(row, header)
            subset_id_to_row[solution_row.subset_id] = solution_row

    return DataEntry(
        solution_file=solution_file,
        subset_id_to_row=subset_id_to_row,
    )


def load_dataset(
    dataset_parameters: DatasetParameters, *, disable_throwaway: bool
) -> Dataset:
    solution_files = get_solution_files(dataset_parameters.working_dir)

    data_entries = []
    print(
        f"Loading dataset from {dataset_parameters.description} with {len(solution_files)} solution files"
    )
    print(f"Dataset {dataset_parameters.description} parameters: {dataset_parameters}")
    thrownaway_count = 0

    for i, solution_file in tqdm(
        enumerate(solution_files),
        desc=f"Loading solution files for {dataset_parameters.description}",
        total=len(solution_files),
    ):
        if not disable_throwaway and (
            i < dataset_parameters.throwaway_first_n
            or i >= len(solution_files) - dataset_parameters.throwaway_last_n
        ):
            thrownaway_count += 1
            continue

        data_entry = load_dataentry(solution_file)
        data_entries.append(data_entry)

    if thrownaway_count > 0:
        print(
            f"Thrown away {thrownaway_count} solution files for dataset {dataset_parameters.description}"
        )

    return Dataset(
        parameters=dataset_parameters,
        data_entries=data_entries,
    )

    
def parallel_load_datasets(
    all_dataset_parameters: list[DatasetParameters],
    *,
    disable_throwaway: bool = False,
) -> list[Dataset]:
    results = [None] * len(all_dataset_parameters)
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = {
            executor.submit(load_dataset, dataset_parameters, disable_throwaway=disable_throwaway): idx
            for idx, dataset_parameters in enumerate(all_dataset_parameters)
        }

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def get_dataset_y_summary_statistics(
    dataset: Dataset,
) -> list[str]:
    output: list[str] = []

    output.append(f"Dataset: {dataset.parameters.description}; images from {dataset.min_image_index} to {dataset.max_image_index} ({len(dataset.data_entries)} images total)")

    NUM_SAMPLES_AT_THE_END = 10
    assert NUM_SAMPLES_AT_THE_END <= len(
        dataset.data_entries
    ), f"Number of samples at the end ({NUM_SAMPLES_AT_THE_END}) cannot be greater than total number of data entries ({len(dataset.data_entries)})"
    displacement_ys = dataset.get_displacement_ys_adjusted(
        start_index=len(dataset.data_entries) - NUM_SAMPLES_AT_THE_END,
    )
    assert len(displacement_ys) == NUM_SAMPLES_AT_THE_END
    displacement_ys = [np.mean(y) for y in displacement_ys]  # Flatten the list of lists

    output.append(
        f"Displacement Y ({MICROMETER}) from the last {NUM_SAMPLES_AT_THE_END} images:"
    )
    displacement_y_min = np.min(displacement_ys)
    displacement_y_max = np.max(displacement_ys)
    displacement_y_p05 = np.percentile(displacement_ys, 5)
    displacement_y_p95 = np.percentile(displacement_ys, 95)
    displacement_y_mean = np.mean(displacement_ys)
    displacement_y_median = np.median(displacement_ys)

    output.append(
        f"Min-Max (diff): {displacement_y_min:2f} - {displacement_y_max:2f} ({displacement_y_max - displacement_y_min:2f})"
    )
    output.append(
        f"P05-P95 (diff): {displacement_y_p05:2f} - {displacement_y_p95:2f} ({displacement_y_p95 - displacement_y_p05:2f})"
    )
    output.append(f"Mean: {displacement_y_mean:2f}; Median: {displacement_y_median:2f}")

    return output


def get_dataset_detailed(dataset: Dataset) -> list[str]:
    ys = dataset.get_displacement_ys_adjusted()
    image_indices = [data_entry.image_index for data_entry in dataset.data_entries]
    means = [np.mean(y) for y in ys]

    output = []
    output.append(
        f"Detailed displacement Y ({MICROMETER}) for dataset: {dataset.parameters.description}"
    )
    for i, (image_index, mean) in enumerate(zip(image_indices, means)):
        line = f"{i} {image_index=} mean_displacement_y_m={mean:.2e}"
        output.append(line)
    return output


def plot_dataset(
    xs: list[float],
    nested_ys: list[list[float]],
    dataset_description: str,
    measurement_label: str,
) -> Figure:
    xs = np.array(xs)
    ys = [np.array(sublist) for sublist in nested_ys]
    means = np.array([np.mean(y) for y in ys])

    fig, ax = plt.subplots()
    ax.plot(
        xs,
        means,
        "-o",
        label=measurement_label,
        color="blue",
        linewidth=0.5,
        markersize=1,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{measurement_label} ({MICROMETER})")
    ax.set_title(f"{measurement_label} over Time: {dataset_description}")
    ax.grid(True)
    return fig


def single_output_file_driver(
    parameters: list[DatasetParameters], analysis_parameters: AnalysisParameters
) -> None:
    t = time.time()
    print(f"Running analysis")
    print(f"Analysis parameters: {analysis_parameters}")
    print(f"Number of datasets: {len(parameters)}")

    output_file_y = PROJECT_ROOT / "analysis_output_displacement_y.pdf"
    output_file_x = PROJECT_ROOT / "analysis_output_displacement_x.pdf"
    output_file_total_displacement = (
        PROJECT_ROOT / "analysis_output_total_displacement.pdf"
    )
    detailed_output_file_y = (
        PROJECT_ROOT / "analysis_output_detailed_displacement_y.txt"
    )

    datasets = [
        load_dataset(param, disable_throwaway=analysis_parameters.disable_throwaway)
        for param in parameters
    ]
    # datasets = parallel_load_datasets(
    #     parameters,
    #     disable_throwaway=analysis_parameters.disable_throwaway,
    # )
    print(f"Loaded {len(datasets)} datasets in {time.time() - t:.2f} seconds")

    summary_statistics_y = [
        get_dataset_y_summary_statistics(dataset) for dataset in datasets
    ]
    plots_y = [
        plot_dataset(
            dataset.data_entry_xs_s,
            dataset.get_displacement_ys_adjusted(),
            dataset.parameters.description,
            "displacement y",
        )
        for dataset in datasets
    ]
    detailed_outputs_y = [
        get_dataset_detailed(dataset)
        for dataset in datasets
        if dataset.parameters.print_detailed
    ]

    plots_x = [
        plot_dataset(
            dataset.data_entry_xs_s,
            dataset.get_displacement_xs_adjusted(),
            dataset.parameters.description,
            "displacement x",
        )
        for dataset in datasets
    ]

    plots_total_displacement = [
        plot_dataset(
            dataset.data_entry_xs_s,
            dataset.get_total_displacements_adjusted(),
            dataset.parameters.description,
            "total displacement",
        )
        for dataset in datasets
    ]

    for (
        output_file,
        detailed_output_file,
        plots,
        summary_statistics,
        detailed_outputs,
    ) in [
        (
            output_file_y,
            detailed_output_file_y,
            plots_y,
            summary_statistics_y,
            detailed_outputs_y,
        ),
        (output_file_x, None, plots_x, [], []),
        (output_file_total_displacement, None, plots_total_displacement, [], []),
    ]:
        with PdfPages(output_file) as pdf:
            fig_summary, ax = plt.subplots(figsize=(8.5, 11))  # A4 size
            ax.axis("off")

            if summary_statistics:
                summary_text = ""
                for i, stats in enumerate(summary_statistics):
                    summary_text += "\n".join(stats) + "\n\n"
                ax.text(
                    0.01, 0.99, summary_text, ha="left", va="top", fontsize=4, wrap=True
                )
                pdf.savefig(fig_summary)
                plt.close(fig_summary)

            for plot in plots:
                pdf.savefig(plot)
                plt.close(plot)

        if detailed_output_file and detailed_outputs:
            with detailed_output_file.open("w") as file:
                for detailed_output in detailed_outputs:
                    for line in detailed_output:
                        file.write(line + "\n")

    print(f"Finished analysis in {time.time() - t:.2f} seconds")


if __name__ == "__main__":
    # Set these filter options to run analysis on a subset of datasets
    # Set all to None to run on all datasets
    filter_labels: Optional[list[ExperimentLabels]] = None
    filter_working_dirs: Optional[list[str]] = None

    single_output_file_driver(
        find_dataset_parameters(labels=filter_labels, working_dirs=filter_working_dirs),
        ANALYSIS_PARAMETERS,
    )
