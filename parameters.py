from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging


class ExperimentLabels(Enum):
    LINER_1_100P = "liner_1_100p"
    LINER_1_80P = "liner_1_80p"
    LINER_1_120P = "liner_1_120p"

    LINER_2_100P = "liner_2_100p"
    LINER_2_80P = "liner_2_80p"
    LINER_2_120P = "liner_2_120p"

    @property
    def pretty_string(self) -> str:
        if self == ExperimentLabels.LINER_1_100P:
            return "Liner 1 $100\%$"
        elif self == ExperimentLabels.LINER_1_80P:
            return "Liner 1 $80\%$"
        elif self == ExperimentLabels.LINER_1_120P:
            return "Liner 1 $120\%$"
        elif self == ExperimentLabels.LINER_2_100P:
            return "Liner 2 $100\%$"
        elif self == ExperimentLabels.LINER_2_80P:
            return "Liner 2 $80\%$"
        elif self == ExperimentLabels.LINER_2_120P:
            return "Liner 2 $120\%$"
        else:
            raise ValueError(f"Cannot convert to string: {self}")


@dataclass
class DatasetParameters:
    working_dir: str
    label: ExperimentLabels

    # Use pixspy.com to get three pixel coordinates of ubolt to compute ubolt diameter in pixels
    ubolt_edge_pixel_coordinate_same_side_a: tuple[int, int]
    ubolt_edge_pixel_coordinate_same_side_b: tuple[int, int]
    ubolt_edge_pixel_coordinate_opposite_side: tuple[int, int]

    throwaway_first_n: int = 0  # Set to throwaway first n images in analysis
    throwaway_last_n: int = 0  # Set to throwaway last n images in analysis
    first_n_mean_as_zero: float = 10  # Use the first 10 image's mean measurement as zero

    print_detailed: bool = (
        False  # Set to output a .txt file with detailed information useful for debugging
    )

    ubolt_diameter_m = 1.5 * 1e-2  # TODO: fill in with real value
    time_interval_per_image_s: float = 10

    @property
    def ubolt_diameter_px(self) -> float:
        p1 = self.ubolt_edge_pixel_coordinate_same_side_a
        p2 = self.ubolt_edge_pixel_coordinate_same_side_b
        p3 = self.ubolt_edge_pixel_coordinate_opposite_side
        twice_area = abs(
            p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
        )
        base_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return twice_area / base_length

    @property
    def frame_y_to_bolt_y_adjustment(self) -> float:
        # Calculate cosine theta of the angle between the line connecting the two same side points and absolute vertical
        hypotenus = (
            (self.ubolt_edge_pixel_coordinate_same_side_a[0] - self.ubolt_edge_pixel_coordinate_same_side_b[0]) ** 2
            + (self.ubolt_edge_pixel_coordinate_same_side_a[1] - self.ubolt_edge_pixel_coordinate_same_side_b[1]) ** 2
        ) ** 0.5
        adjacent = abs(self.ubolt_edge_pixel_coordinate_same_side_a[1] - self.ubolt_edge_pixel_coordinate_same_side_b[1])
        return adjacent / hypotenus

    @property
    def px_to_m(self) -> float:
        return self.ubolt_diameter_m / self.ubolt_diameter_px

    @property
    def description(self) -> float:
        return self.label.value + "(" + self.working_dir + ")"


ALL_DATASET_PARAMETERS = sorted(
    [
        DatasetParameters(
            working_dir="0601_WorkingDir",
            label=ExperimentLabels.LINER_1_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(2966, 3632),
            ubolt_edge_pixel_coordinate_same_side_b=(2964, 3777),
            ubolt_edge_pixel_coordinate_opposite_side=(4066, 3362),
            throwaway_last_n=10,
        ),
        DatasetParameters(
            working_dir="0614_WorkingDir",
            label=ExperimentLabels.LINER_1_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(2894, 3377),
            ubolt_edge_pixel_coordinate_same_side_b=(2908, 2871),
            ubolt_edge_pixel_coordinate_opposite_side=(3937, 2646),
        ),
        DatasetParameters(
            working_dir="0615_WorkingDir",
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(2595, 3206),
            ubolt_edge_pixel_coordinate_same_side_b=(2562, 4275),
            ubolt_edge_pixel_coordinate_opposite_side=(3699, 3186),
        ),
        DatasetParameters(
            working_dir="0619_WorkingDir",
            label=ExperimentLabels.LINER_1_80P,
            throwaway_first_n=1,
            ubolt_edge_pixel_coordinate_same_side_a=(3303, 3506),
            ubolt_edge_pixel_coordinate_same_side_b=(3301, 3157),
            ubolt_edge_pixel_coordinate_opposite_side=(4268, 3202),
        ),
        DatasetParameters(
            working_dir="0620_WorkingDir",
            label=ExperimentLabels.LINER_1_100P,
            throwaway_first_n=1,
            ubolt_edge_pixel_coordinate_same_side_a=(2966, 3523),
            ubolt_edge_pixel_coordinate_same_side_b=(2965, 3762),
            ubolt_edge_pixel_coordinate_opposite_side=(4053, 3388),
        ),
        DatasetParameters(
            working_dir="0620Midnight_WorkingDir",
            label=ExperimentLabels.LINER_1_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(3174, 3750),
            ubolt_edge_pixel_coordinate_same_side_b=(3169, 3429),
            ubolt_edge_pixel_coordinate_opposite_side=(4203, 3378),
        ),
        DatasetParameters(
            working_dir = '0621_WorkingDir',
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(2640, 3655),
            ubolt_edge_pixel_coordinate_same_side_b=(2625, 4381),
            ubolt_edge_pixel_coordinate_opposite_side=(3758, 4154),
        ),
        DatasetParameters(
            working_dir="0621Midnight_WorkingDir", 
            label=ExperimentLabels.LINER_1_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(3042, 3731),
            ubolt_edge_pixel_coordinate_same_side_b=(3031, 4123),
            ubolt_edge_pixel_coordinate_opposite_side=(4188, 3542),
        ),
        DatasetParameters(
            working_dir="0622_WorkingDir",
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(2522, 3514),
            ubolt_edge_pixel_coordinate_same_side_b=(2507, 4197),
            ubolt_edge_pixel_coordinate_opposite_side=(3631, 3773),
        ),
        DatasetParameters(
            working_dir="0628_WorkingDir",
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(3650, 3629),
            ubolt_edge_pixel_coordinate_same_side_b=(3620, 4282),
            ubolt_edge_pixel_coordinate_opposite_side=(4783, 3801),
        ),
        DatasetParameters(
            working_dir="0628Dinner_WorkingDir",
            label=ExperimentLabels.LINER_1_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(3030, 3562),
            ubolt_edge_pixel_coordinate_same_side_b=(3011, 4244),
            ubolt_edge_pixel_coordinate_opposite_side=(4086, 3909),
        ),
        DatasetParameters(
            working_dir="0628Midnight_WorkingDir",
            label=ExperimentLabels.LINER_2_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(3237, 3459),
            ubolt_edge_pixel_coordinate_same_side_b=(3228, 3957),
            ubolt_edge_pixel_coordinate_opposite_side=(4337, 821),
        ),
        DatasetParameters(
            working_dir="0629_WorkingDir",
            label=ExperimentLabels.LINER_2_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(2544, 3503),
            ubolt_edge_pixel_coordinate_same_side_b=(2531, 4352),
            ubolt_edge_pixel_coordinate_opposite_side=(3564, 3545),
        ),
        DatasetParameters(
            working_dir="0704Dinner_WorkingDir",
            label=ExperimentLabels.LINER_2_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(3058, 3988),
            ubolt_edge_pixel_coordinate_same_side_b=(3044, 4494),
            ubolt_edge_pixel_coordinate_opposite_side=(4071, 4252),
        ),
        DatasetParameters(
            working_dir="0704Midnight_WorkingDir",
            label=ExperimentLabels.LINER_2_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(2940, 4327),
            ubolt_edge_pixel_coordinate_same_side_b=(2952, 3987),
            ubolt_edge_pixel_coordinate_opposite_side=(3957, 4301),
        ),
        DatasetParameters(
            working_dir="0705_WorkingDir",
            label=ExperimentLabels.LINER_2_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(1912, 4175),
            ubolt_edge_pixel_coordinate_same_side_b=(1901, 4392),
            ubolt_edge_pixel_coordinate_opposite_side=(2925, 4288),
        ),
        DatasetParameters(
            working_dir="0705Dinner_WorkingDir",
            label=ExperimentLabels.LINER_2_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(3958, 3663),
            ubolt_edge_pixel_coordinate_same_side_b=(3954, 3810),
            ubolt_edge_pixel_coordinate_opposite_side=(4787, 3798),
        ),
        DatasetParameters(
            working_dir="0706_WorkingDir",
            label=ExperimentLabels.LINER_2_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(3506, 3732),
            ubolt_edge_pixel_coordinate_same_side_b=(3500, 4080),
            ubolt_edge_pixel_coordinate_opposite_side=(4328, 3991),
        ),
        DatasetParameters(
            working_dir="0706Dinner_WorkingDir",
            label=ExperimentLabels.LINER_2_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(3539, 4157),
            ubolt_edge_pixel_coordinate_same_side_b=(3530, 4580),
            ubolt_edge_pixel_coordinate_opposite_side=(4567, 4292),
        ),
        DatasetParameters(
            working_dir="0707_WorkingDir",
            label=ExperimentLabels.LINER_2_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(3590, 4143),
            ubolt_edge_pixel_coordinate_same_side_b=(3579, 4594),
            ubolt_edge_pixel_coordinate_opposite_side=(4607, 4357),
        ),
        DatasetParameters(
            working_dir="0712Midnight_WorkingDir",
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(3255, 3516),
            ubolt_edge_pixel_coordinate_same_side_b=(3237, 4166),
            ubolt_edge_pixel_coordinate_opposite_side=(4149, 3485),
        ),
        DatasetParameters(
            working_dir="0725_WorkingDir",
            label=ExperimentLabels.LINER_1_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(3057, 3593),
            ubolt_edge_pixel_coordinate_same_side_b=(3041, 4273),
            ubolt_edge_pixel_coordinate_opposite_side=(4146, 3965),
        ),
        DatasetParameters(
            working_dir="0726_WorkingDir",
            label=ExperimentLabels.LINER_2_120P,
            ubolt_edge_pixel_coordinate_same_side_a=(2971, 3595),
            ubolt_edge_pixel_coordinate_same_side_b=(2953, 4123),
            ubolt_edge_pixel_coordinate_opposite_side=(4067, 3726),
        ),
        DatasetParameters(
            working_dir="0726Midnight_WorkingDir",
            label=ExperimentLabels.LINER_1_80P,
            ubolt_edge_pixel_coordinate_same_side_a=(2503, 3499),
            ubolt_edge_pixel_coordinate_same_side_b=(2473, 4423),
            ubolt_edge_pixel_coordinate_opposite_side=(3595, 3821),
        ),
        DatasetParameters(
            working_dir="0727_WorkingDir",
            label=ExperimentLabels.LINER_1_100P,
            ubolt_edge_pixel_coordinate_same_side_a=(2396, 3552),
            ubolt_edge_pixel_coordinate_same_side_b=(2371, 4307),
            ubolt_edge_pixel_coordinate_opposite_side=(3490, 3862),
        ),
    ],
    key=lambda x: x.label.value,
)


@dataclass
class AnalysisParameters:
    disable_throwaway: bool  # True to disable throwaway settings to see all data


ANALYSIS_PARAMETERS = AnalysisParameters(disable_throwaway=False)


def find_dataset_parameters(
    *, labels: Optional[ExperimentLabels] = [], working_dirs: Optional[list[str]] = []
) -> list[DatasetParameters]:
    parameters = ALL_DATASET_PARAMETERS

    if labels is not None:
        assert len(labels) > 0, "Labels must not be empty"
        parameters = [p for p in parameters if p.label in labels]

    if working_dirs is not None:
        assert len(working_dirs) > 0, "Working directories must not be empty"
        parameters = [p for p in parameters if p.working_dir in working_dirs]

    assert (
        len(parameters) > 0
    ), f"No dataset parameters found for the filter options: labels={labels}, working_dirs={working_dirs}"
    return parameters


if __name__ == "__main__":
    for dataset in ALL_DATASET_PARAMETERS:
        logging.info(
            f"{dataset.description}: ubolt_diameter_px={dataset.ubolt_diameter_px:.2f}, px_to_m={dataset.px_to_m:.6e}"
        )