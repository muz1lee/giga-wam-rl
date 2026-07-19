import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.gwp05_pilot_inputs import (  # noqa: E402
    build_action_state_conditions,
    compose_t_layout,
    cover_resize_crop_geometry,
    make_additive_action_counterfactual,
    rgb_frames_to_vae_video,
)


class GWP05PilotInputTests(unittest.TestCase):
    def test_center_crop_geometry_matches_official_open_loop_layout(self) -> None:
        self.assertEqual(
            cover_resize_crop_geometry(
                source_width=320,
                source_height=240,
                target_width=320,
                target_height=192,
            ),
            (320, 240, 0, 24),
        )
        self.assertEqual(
            cover_resize_crop_geometry(
                source_width=320,
                source_height=240,
                target_width=160,
                target_height=192,
            ),
            (256, 192, 48, 0),
        )

    def test_three_rgb_views_form_the_384_by_320_t_layout(self) -> None:
        front = np.full((240, 320, 3), (255, 0, 0), dtype=np.uint8)
        left = np.full((240, 320, 3), (0, 255, 0), dtype=np.uint8)
        right = np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8)

        composite = compose_t_layout(front, left, right)

        self.assertEqual(composite.shape, (384, 320, 3))
        np.testing.assert_array_equal(composite[96, 160], (255, 0, 0))
        np.testing.assert_array_equal(composite[288, 80], (0, 255, 0))
        np.testing.assert_array_equal(composite[288, 240], (0, 0, 255))

    def test_action_condition_is_delta_for_joints_and_absolute_for_grippers(self) -> None:
        state = np.arange(14, dtype=np.float32)
        actions = np.tile(state + 2.0, (48, 1))
        delta_mask = np.array(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=bool
        )
        state_stats = {
            "q01": np.zeros(16, dtype=np.float32),
            "q99": np.full(16, 20.0, dtype=np.float32),
        }
        action_stats = {
            "q01": np.zeros(16, dtype=np.float32),
            "q99": np.full(16, 20.0, dtype=np.float32),
        }

        normalized_state, normalized_action = build_action_state_conditions(
            state,
            actions,
            state_stats=state_stats,
            action_stats=action_stats,
            delta_mask=delta_mask,
            model_dimensions=16,
        )

        self.assertEqual(normalized_state.shape, (1, 16))
        self.assertEqual(normalized_action.shape, (48, 16))
        np.testing.assert_allclose(normalized_action[:, :6], -0.8)
        np.testing.assert_allclose(normalized_action[:, 7:13], -0.8)
        self.assertAlmostEqual(float(normalized_action[0, 6]), -0.2)
        self.assertAlmostEqual(float(normalized_action[0, 13]), 0.5)
        np.testing.assert_array_equal(normalized_state[:, 14:], 0.0)
        np.testing.assert_array_equal(normalized_action[:, 14:], 0.0)

    def test_rgb_frames_are_converted_to_bcthw_minus_one_to_one(self) -> None:
        frame0 = np.zeros((384, 320, 3), dtype=np.uint8)
        frame1 = np.full((384, 320, 3), 255, dtype=np.uint8)

        video = rgb_frames_to_vae_video([frame0, frame1])

        self.assertEqual(tuple(video.shape), (1, 3, 2, 384, 320))
        self.assertEqual(float(video[:, :, 0].min()), -1.0)
        self.assertEqual(float(video[:, :, 1].max()), 1.0)

    def test_counterfactual_changes_only_the_requested_physical_dimension(self) -> None:
        actions = np.arange(48 * 14, dtype=np.float32).reshape(48, 14)

        perturbed = make_additive_action_counterfactual(
            actions, action_dimension=3, additive_offset=0.5
        )

        self.assertFalse(np.shares_memory(actions, perturbed))
        np.testing.assert_array_equal(perturbed[:, :3], actions[:, :3])
        np.testing.assert_allclose(perturbed[:, 3], actions[:, 3] + 0.5)
        np.testing.assert_array_equal(perturbed[:, 4:], actions[:, 4:])


if __name__ == "__main__":
    unittest.main()
