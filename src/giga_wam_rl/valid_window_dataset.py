from typing import Any

from giga_datasets.datasets.dataset import register_dataset
from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset

from giga_wam_rl.valid_windows import (
    select_valid_window_indices,
    valid_window_indices,
)


def _metadata_episode_lengths(metadata: Any) -> list[int]:
    episodes = metadata.episodes
    lengths = []
    for index in range(len(episodes)):
        row = episodes[index]
        if "length" in row:
            lengths.append(int(row["length"]))
        elif "dataset_from_index" in row and "dataset_to_index" in row:
            lengths.append(
                int(row["dataset_to_index"]) - int(row["dataset_from_index"])
            )
        else:
            raise KeyError("LeRobot episode metadata has no length or index bounds")
    return lengths


@register_dataset
class ValidWindowWAMLeRobotDataset(WAMLeRobotDataset):
    """Expose only starts with a complete action and future-observation horizon."""

    def __init__(
        self,
        valid_horizon: int,
        selected_raw_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.valid_horizon = int(valid_horizon)
        self.selected_raw_indices = selected_raw_indices
        self._valid_indices: list[int] | None = None

    def open(self) -> None:
        super().open()
        if self._valid_indices is None:
            all_valid_indices = valid_window_indices(
                _metadata_episode_lengths(self.dataset.meta),
                horizon=self.valid_horizon,
            )
            self._valid_indices = (
                all_valid_indices
                if self.selected_raw_indices is None
                else select_valid_window_indices(
                    all_valid_indices, self.selected_raw_indices
                )
            )

    def __len__(self) -> int:
        self.open()
        return len(self._valid_indices)

    def _get_data(self, index: int) -> dict:
        if self._valid_indices is None:
            raise RuntimeError("dataset must be opened before indexing")
        return super()._get_data(self._valid_indices[index])

    def close(self) -> None:
        self._valid_indices = None
        super().close()
