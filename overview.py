from parameters import ALL_DATASET_PARAMETERS, ExperimentLabels
from dataclasses import dataclass
from typing import Optional, Union, List
from tabulate import tabulate
from collections import defaultdict
import logging


@dataclass
class DatasetOverview:
    working_dir: str

    @property
    def label(self) -> ExperimentLabels:
        matching_dataset_parameters = [
            p for p in ALL_DATASET_PARAMETERS if p.working_dir == self.working_dir
        ]
        if not matching_dataset_parameters:
            raise ValueError(f"No dataset parameters found for working_dir: {self.working_dir}")
        if len(matching_dataset_parameters) > 1:
            raise ValueError(f"Multiple dataset parameters found for working_dir: {self.working_dir}")
        return matching_dataset_parameters[0].label

    y_displacement_shape: str
    steady_state_value: Optional[Union[float, str]]
    steady_state_x_value: Optional[Union[float, str]]

    x_displacement_shape: str
    total_displacement_shape: str


ALL_DATASET_OVERVIEWS = [
    DatasetOverview(    
        working_dir="0601_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-47,
        steady_state_x_value=10000,
        x_displacement_shape="slip build up 0 to -15",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0614_WorkingDir",
        y_displacement_shape="has_two_plateaus",
        steady_state_value="-47 and -57",
        steady_state_x_value=6000,
        x_displacement_shape="nike swoosh with slip at pleateau boundary min -30",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0615_WorkingDir",
        y_displacement_shape="rapid goes down",
        steady_state_value=-120,
        steady_state_x_value=12000,
        x_displacement_shape="exponential decayed increase to 28",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0619_WorkingDir",
        y_displacement_shape="good but slightly goes down towards end",
        steady_state_value=-45,
        steady_state_x_value=8000,
        x_displacement_shape="exponential decayed increase to 20, then plateau, slightly goes down at the end",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0620_WorkingDir",
        y_displacement_shape="was_good_but_became_linear",
        steady_state_value=-45,
        steady_state_x_value=7000,
        x_displacement_shape="exponential decayed increase to 80",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0620Midnight_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-40,
        steady_state_x_value=8000,
        x_displacement_shape="nike swoosh -5 to 5",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0621_WorkingDir",
        y_displacement_shape="weird sine",
        steady_state_value="-5 to -20",
        steady_state_x_value=2500,
        x_displacement_shape="horizontal noise, slight dip at the beginning 5 to -10",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0621Midnight_WorkingDir",
        y_displacement_shape="good but a bit slow",
        steady_state_value=-27,
        steady_state_x_value=12500,
        x_displacement_shape="exponential decayed increase to 35",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0622_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-33,
        steady_state_x_value=7500,
        x_displacement_shape="exponential decayed increase to 18",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0628_WorkingDir",
        y_displacement_shape="weird convex up",
        steady_state_value="none",
        steady_state_x_value="none",
        x_displacement_shape="slight dip then, slowly increasing from -5 to above 0",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0628Dinner_WorkingDir",
        y_displacement_shape="good but a bit slow",
        steady_state_value=-30,
        steady_state_x_value=9000,
        x_displacement_shape="exponential decayed increase to 35",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0628Midnight_WorkingDir",
        y_displacement_shape="good but curve up towards the far end",
        steady_state_value=-26,
        steady_state_x_value=7500,
        x_displacement_shape="exponential decayed increase to 5-10 then kink down towards the end",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0629_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-18,
        steady_state_x_value=3000,
        x_displacement_shape="exponential decayed increase to 40",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0704Dinner_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-20,
        steady_state_x_value=2000,
        x_displacement_shape="slight exponential decayed increase to 13",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0704Midnight_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-20,
        steady_state_x_value=2000,
        x_displacement_shape="exponential decayed decrase to -20",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0705_WorkingDir",
        y_displacement_shape="super weird positive",
        steady_state_value=5,
        steady_state_x_value=6000,
        x_displacement_shape="horizontal noise at -5",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0705Dinner_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-32,
        steady_state_x_value=2500,
        x_displacement_shape="slight exponential decayed increase to 25",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0706_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-28,
        steady_state_x_value=3000,
        x_displacement_shape="kind of linear increase to 18",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0706Dinner_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-28,
        steady_state_x_value=3000,
        x_displacement_shape="exponential decayed decrease to -30, slightly linear plateau until -35",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0707_WorkingDir",
        y_displacement_shape="good",
        steady_state_value=-28,
        steady_state_x_value=3500,
        x_displacement_shape="exponential decayed increase to 18",
        total_displacement_shape="good",
    ),
    DatasetOverview(    
        working_dir="0712Midnight_WorkingDir",
        y_displacement_shape="good but very far out starts new plateau",
        steady_state_value="-27 then -40",
        steady_state_x_value=8000,
        x_displacement_shape="exponential decayed increase to 100",
        total_displacement_shape="good",
    ),

    DatasetOverview(    
        working_dir="0725_WorkingDir",
        y_displacement_shape="good, a bit noisy, far out starts nike swoosh",
        steady_state_value="-33",
        steady_state_x_value=8000,
        x_displacement_shape="5 to 10 wavy",
        total_displacement_shape="reaches 35 peak then reverse nike swoosh",
    ),
    DatasetOverview(    
        working_dir="0726_WorkingDir",
        y_displacement_shape="starts of good, but raises up a bit",
        steady_state_value="-18 then -15",
        steady_state_x_value=5000,
        x_displacement_shape="linear decrease to -15/-20",
        total_displacement_shape="good to 20",
    ),
    DatasetOverview(    
        working_dir="0726Midnight_WorkingDir",
        y_displacement_shape="good",
        steady_state_value="-33",
        steady_state_x_value=8000,
        x_displacement_shape="exponential decrease to -15",
        total_displacement_shape="good to 38",
    ),
    DatasetOverview(    
        working_dir="0727_WorkingDir",
        y_displacement_shape="good",
        steady_state_value="-23",
        steady_state_x_value=7500,
        x_displacement_shape="noise -5 to 5 horizontal",
        total_displacement_shape="good to 22",
    ),
]

def export_dataset_overview_table(datasets: List[DatasetOverview], output_path: str):
    # Group datasets by label
    grouped = defaultdict(list)
    for dataset in datasets:
        grouped[dataset.label].append(dataset)

    # Prepare table rows
    table_rows = []
    prev_label = None
    for label, dataset_group in grouped.items():
        for ds in dataset_group:
            table_rows.append([
                str(label) if label != prev_label else "",  # Only show label once per group
                ds.working_dir,
                ds.y_displacement_shape,
                ds.steady_state_value if ds.steady_state_value is not None else "N/A",
                ds.steady_state_x_value if ds.steady_state_x_value is not None else "N/A",
                ds.x_displacement_shape,
                # ds.total_displacement_shape,
            ])
            prev_label = label

    # Column headers
    headers = [
        "Label",
        "Working Dir",
        "Y Displacement Shape",
        "Steady State Y",
        "Steady State X",
        "X Displacement Shape",
        # "Total Displacement Shape",
    ]

    # Format ASCII table
    table = tabulate(table_rows, headers=headers, tablefmt="grid")

    # Write to file
    with open(output_path, "w") as f:
        f.write(table)

    logging.info(f"Dataset overview exported to: {output_path}")

    
if __name__ == "__main__":
    for dp in ALL_DATASET_PARAMETERS:
        if dp.working_dir not in [d.working_dir for d in ALL_DATASET_OVERVIEWS]:
            logging.error(f"No overview entry for working_dir: {dp.working_dir} (label: {dp.label})")

    export_dataset_overview_table(ALL_DATASET_OVERVIEWS, "dataset_overview.txt")