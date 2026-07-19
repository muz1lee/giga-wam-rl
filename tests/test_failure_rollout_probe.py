import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.failure_rollout_probe import (  # noqa: E402
    _load_observation_chunk,
    _observation_files_from_sidecar,
    causal_action_rows,
    parse_obs_chunk_index,
    stitch_observation_chunks,
)


def _observation(value: float) -> dict[str, object]:
    pixel = np.uint8(value)
    return {
        "observation.images.cam_high": np.full((2, 3, 3), pixel, dtype=np.uint8),
        "observation.images.cam_left_wrist": np.full((2, 3, 3), pixel, dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((2, 3, 3), pixel, dtype=np.uint8),
        "observation.state": np.full(14, value, dtype=np.float64),
        "task": "place bread",
    }


class FailureRolloutProbeTests(unittest.TestCase):
    def test_sidecar_archive_closes_observation_file_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "server_visualization" / "obs_data_0.pt"
            archive = root / "negative" / "tensors" / "obs_data_0.pt"
            source.parent.mkdir(parents=True)
            archive.parent.mkdir(parents=True)
            source.touch()
            archive.symlink_to(source)
            sidecar_row = {
                "files": [
                    {
                        "name": "obs_data_0.pt",
                        "source": str(source),
                        "path": str(archive),
                        "mode": "symlink",
                    }
                ]
            }

            paths = _observation_files_from_sidecar(
                sidecar_row,
                asset_root=root,
                read_only_roots=[root.resolve()],
            )

        self.assertEqual(paths, [archive])

    def test_numpy_observation_chunk_loads_with_weights_only(self) -> None:
        import torch

        observation = {
            "observation.images.cam_high": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.images.cam_left_wrist": np.zeros(
                (240, 320, 3), dtype=np.uint8
            ),
            "observation.images.cam_right_wrist": np.zeros(
                (240, 320, 3), dtype=np.uint8
            ),
            "observation.state": np.zeros(14, dtype=np.float64),
            "task": "place bread",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "obs_data_0.pt"
            torch.save([observation], path)

            loaded = _load_observation_chunk(path)

        self.assertEqual(len(loaded), 1)
        np.testing.assert_array_equal(loaded[0]["observation.state"], np.zeros(14))

    def test_chunk_index_uses_numeric_order(self) -> None:
        paths = [Path("obs_data_10.pt"), Path("obs_data_2.pt")]

        self.assertEqual(
            sorted(paths, key=parse_obs_chunk_index),
            [Path("obs_data_2.pt"), Path("obs_data_10.pt")],
        )
        with self.assertRaises(ValueError):
            parse_obs_chunk_index(Path("actions_2.pt"))

    def test_stitch_handles_cumulative_and_boundary_overlap(self) -> None:
        a, b, c, d, e = [_observation(value) for value in range(5)]

        stitched, overlaps = stitch_observation_chunks(
            [[a, b, c], [a, b, c, d], [d, e]]
        )

        self.assertEqual(
            [row["observation.state"][0] for row in stitched], [0, 1, 2, 3, 4]
        )
        self.assertEqual(overlaps, [3, 1])

    def test_causal_rows_use_next_drive_target_and_drop_terminal_duplicates(
        self,
    ) -> None:
        observations = [_observation(value) for value in [0, 1, 2, 2, 2]]

        rows, summary = causal_action_rows(observations, drop_terminal_duplicates=True)

        self.assertEqual(len(rows), 3)
        np.testing.assert_array_equal(rows[0]["observation.state"], np.zeros(14))
        np.testing.assert_array_equal(rows[0]["action"], np.ones(14))
        np.testing.assert_array_equal(rows[-1]["action"], np.full(14, 2.0))
        self.assertEqual(summary["input_observations"], 5)
        self.assertEqual(summary["terminal_duplicates_dropped"], 2)
        self.assertEqual(summary["causal_rows"], 3)


if __name__ == "__main__":
    unittest.main()
