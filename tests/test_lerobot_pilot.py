import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.lerobot_pilot import _features  # noqa: E402


class LeRobotPilotTests(unittest.TestCase):
    def test_features_preserve_physical_14d_vectors_and_three_rgb_views(self) -> None:
        action_names = [f"joint_{index}" for index in range(14)]

        features = _features(action_names)

        self.assertEqual(features["observation.state"]["shape"], (14,))
        self.assertEqual(features["action"]["shape"], (14,))
        self.assertEqual(features["observation.state"]["names"], action_names)
        self.assertEqual(features["action"]["names"], action_names)
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            feature = features[f"observation.images.{camera}"]
            self.assertEqual(feature["dtype"], "video")
            self.assertEqual(feature["shape"], (3, 240, 320))


if __name__ == "__main__":
    unittest.main()
