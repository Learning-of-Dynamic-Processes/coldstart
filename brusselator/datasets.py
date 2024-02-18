from multiprocessing import Pool

import numpy as np
import torch
from pydszoo import brusselator  # pylint: disable=E0401
from tqdm.auto import tqdm

from brusselator.config import config


class BrusselatorDataset:
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, num_trajectories: int, len_trajectories: int, parameters: dict) -> None:
        """Create set of trajectories."""
        time_array = np.linspace(0, len_trajectories / 5, len_trajectories + 1)
        self.ids = np.arange(num_trajectories)

        self.input_data = []
        self.output_data = []
        self.v_data = []
        for _ in tqdm(range(num_trajectories), leave=True, position=0):
            initial_condition = brusselator.create_initial_condition()
            trajectory = brusselator.integrate(initial_condition, time_array, parameters)
            self.input_data.append(trajectory[:-1, :1])
            self.output_data.append(trajectory[1:, :1])
            self.v_data.append(trajectory[:-1, 1:])

        self.input_data = np.array(self.input_data)
        self.output_data = np.array(self.output_data)
        self.v_data = np.array(self.v_data)
        self.tt = time_array

    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple:
        """Return a trajectory."""
        return torch.tensor(self.input_data[self.ids[index]], dtype=config["TRAINING"]["dtype"]), torch.tensor(
            self.output_data[self.ids[index]], dtype=config["TRAINING"]["dtype"]
        )

    def save_data(self, path: str, filename: str) -> None:
        """Save the trajectories."""
        np.savez(
            path + filename,
            input_data=self.input_data,
            output_data=self.output_data,
            v_data=self.v_data,
            tt_arr=self.tt_arr,
            ids=self.ids,
        )


def create_trajectory(inputs):
    time_array, parameters, initial_condition = inputs
    trajectory = brusselator.integrate(initial_condition, time_array, parameters)
    return trajectory[:-1, :1], trajectory[1:, :1], trajectory[:-1, 1:]


class BrusselatorParallelDataset:
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, num_trajectories: int, len_trajectories: int, parameters: dict) -> None:
        """Create set of trajectories."""
        time_array = np.linspace(0, len_trajectories / 5, len_trajectories + 1)
        self.ids = np.arange(num_trajectories)

        with Pool(processes=4) as p:
            self.input_data, self.output_data, self.v_data = list(
                zip(
                    *p.map(
                        create_trajectory,
                        [
                            [time_array, parameters, brusselator.create_initial_condition()]
                            for _ in range(num_trajectories)
                        ],
                    )
                )
            )

        self.input_data = np.array(self.input_data)
        self.output_data = np.array(self.output_data)
        self.v_data = np.array(self.v_data)
        self.tt = time_array

    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple:
        """Return a trajectory."""
        return torch.tensor(self.input_data[self.ids[index]], dtype=config["TRAINING"]["dtype"]), torch.tensor(
            self.output_data[self.ids[index]], dtype=config["TRAINING"]["dtype"]
        )

    def save_data(self, path: str, filename: str) -> None:
        """Save the trajectories."""
        np.savez(
            path + filename,
            input_data=self.input_data,
            output_data=self.output_data,
            v_data=self.v_data,
            tt_arr=self.tt_arr,
            ids=self.ids,
        )
