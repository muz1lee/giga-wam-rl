import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.gwp05_smoke import (  # noqa: E402
    ContractError,
    expected_timestep_tokens,
    expected_visual_output_shapes,
    validate_loading_info,
    validate_model_config,
)


class GWP05ContractTests(unittest.TestCase):
    def test_accepts_pinned_checkpoint_config(self) -> None:
        validate_model_config(
            {
                "_class_name": "CasualWorldActionTransformer_MoT",
                "in_channels": 48,
                "out_channels": 48,
                "in_action_channels": 16,
                "out_action_channels": 16,
                "num_layers": 30,
                "action_expert_dim": 1024,
                "num_embodiments": 2,
            }
        )

    def test_rejects_stale_32d_action_contract(self) -> None:
        with self.assertRaisesRegex(ContractError, "in_action_channels"):
            validate_model_config(
                {
                    "_class_name": "CasualWorldActionTransformer_MoT",
                    "in_channels": 48,
                    "out_channels": 48,
                    "in_action_channels": 32,
                    "out_action_channels": 16,
                    "num_layers": 30,
                    "action_expert_dim": 1024,
                    "num_embodiments": 2,
                }
            )

    def test_realistic_joint_forward_uses_289_timesteps(self) -> None:
        self.assertEqual(expected_timestep_tokens(), 289)

    def test_visual_output_contains_reference_and_future_latents(self) -> None:
        full_shape, future_shape = expected_visual_output_shapes()

        self.assertEqual(full_shape, (1, 48, 2, 24, 20))
        self.assertEqual(future_shape, (1, 48, 1, 24, 20))

    def test_loading_info_must_be_strictly_empty(self) -> None:
        validate_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )

        with self.assertRaisesRegex(ContractError, "missing_keys"):
            validate_loading_info(
                {
                    "missing_keys": ["action_encoder.in_proj.weight"],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
