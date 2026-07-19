import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.gwp05_future_sampler import (  # noqa: E402
    FUTURE_LATENT_SHAPE,
    VISUAL_FLOW_SHIFT,
    build_future_only_timestep,
)


class GWP05FutureSamplerTests(unittest.TestCase):
    def test_future_only_timestep_keeps_state_ref_and_action_clean(self) -> None:
        timestep = build_future_only_timestep(
            batch_size=2,
            visual_timestep=750,
            num_state_tokens=1,
            num_action_tokens=48,
            num_ref_tokens=120,
            num_future_tokens=120,
        )

        self.assertEqual(len(timestep), 2)
        self.assertTrue(all(len(row) == 289 for row in timestep))
        for row in timestep:
            self.assertEqual(row[: 1 + 120 + 48], [0] * 169)
            self.assertEqual(row[169:], [750] * 120)

    def test_sampler_contract_matches_visual_training_shift_and_shape(self) -> None:
        self.assertEqual(VISUAL_FLOW_SHIFT, 2.0)
        self.assertEqual(FUTURE_LATENT_SHAPE, (48, 1, 24, 20))

    def test_invalid_token_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_future_only_timestep(
                batch_size=1,
                visual_timestep=500,
                num_state_tokens=1,
                num_action_tokens=48,
                num_ref_tokens=0,
                num_future_tokens=120,
            )


if __name__ == "__main__":
    unittest.main()
