import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.raw_hdf5_probe import (  # noqa: E402
    candidate_window_count,
    sample_indices,
    shifted_action_indices,
)


class RawHDF5ProbeTests(unittest.TestCase):
    def test_candidate_count_respects_one_step_action_shift(self) -> None:
        self.assertEqual(candidate_window_count(160, horizon=48), 112)
        self.assertEqual(candidate_window_count(48, horizon=48), 0)

    def test_sample_indices_align_action_and_future_causally(self) -> None:
        indices = sample_indices(10, horizon=48, frame_offsets=(0, 12, 24, 36, 48))

        self.assertEqual(indices["state"], 10)
        self.assertEqual(indices["actions"], list(range(11, 59)))
        self.assertEqual(indices["images"], [10, 22, 34, 46, 58])

    def test_invalid_alignment_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            candidate_window_count(160, horizon=0)
        with self.assertRaises(ValueError):
            sample_indices(0, horizon=48, frame_offsets=(0, 12, 24, 36, 47))

    def test_lerobot_rows_store_the_next_drive_target(self) -> None:
        self.assertEqual(shifted_action_indices(4), [1, 2, 3, 3])
        with self.assertRaises(ValueError):
            shifted_action_indices(0)


if __name__ == "__main__":
    unittest.main()
