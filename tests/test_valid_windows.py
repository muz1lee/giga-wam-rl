import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.valid_windows import (  # noqa: E402
    select_valid_window_indices,
    valid_window_indices,
)


class ValidWindowTests(unittest.TestCase):
    def test_indices_never_enter_episode_terminal_padding(self) -> None:
        indices = valid_window_indices([5, 7], horizon=3)

        self.assertEqual(indices, [0, 1, 5, 6, 7, 8])

    def test_failure_pilot_has_762_valid_starts(self) -> None:
        indices = valid_window_indices([175] * 6, horizon=48)

        self.assertEqual(len(indices), 762)
        self.assertEqual(indices[:2], [0, 1])
        self.assertEqual(indices[-1], 6 * 175 - 49)

    def test_non_positive_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            valid_window_indices([10], horizon=0)

    def test_explicit_overfit_starts_must_be_valid_and_unique(self) -> None:
        valid = valid_window_indices([10], horizon=3)

        self.assertEqual(select_valid_window_indices(valid, [0, 3, 6]), [0, 3, 6])
        with self.assertRaises(ValueError):
            select_valid_window_indices(valid, [0, 7])
        with self.assertRaises(ValueError):
            select_valid_window_indices(valid, [0, 0])


if __name__ == "__main__":
    unittest.main()
