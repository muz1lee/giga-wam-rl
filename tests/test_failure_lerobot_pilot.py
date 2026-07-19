import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.failure_lerobot_pilot import (  # noqa: E402
    ACTION_NAMES,
    _camera_frame_mae,
    _registered_asset,
)


class FailureLeRobotPilotTests(unittest.TestCase):
    def test_action_schema_matches_14d_drive_target(self) -> None:
        self.assertEqual(len(ACTION_NAMES), 14)
        self.assertEqual(ACTION_NAMES[6], "left_gripper")
        self.assertEqual(ACTION_NAMES[-1], "right_gripper")

    def test_pilot_config_selects_six_unique_failures(self) -> None:
        config_path = (
            PROJECT_ROOT / "configs" / "datasets" / "place_bread_failure_pilot.toml"
        )
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        seeds = [episode["seed"] for episode in config["episodes"]]
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)
        self.assertEqual(config["probe"]["action_horizon"], 48)

    def test_source_asset_must_be_student_read_only(self) -> None:
        registry = {
            "assets": [
                {
                    "name": "failures",
                    "owner": "student",
                    "read_only": False,
                }
            ]
        }

        with self.assertRaises(ValueError):
            _registered_asset(registry, "failures")

    def test_camera_validation_rejects_large_conversion_error(self) -> None:
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        converted = np.ones((2, 3, 3), dtype=np.uint8)

        self.assertEqual(_camera_frame_mae(converted, source, maximum_mae=2.0), 1.0)
        with self.assertRaises(ValueError):
            _camera_frame_mae(converted * 10, source, maximum_mae=2.0)


if __name__ == "__main__":
    unittest.main()
