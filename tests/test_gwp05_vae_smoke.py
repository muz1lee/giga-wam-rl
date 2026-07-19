import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.gwp05_vae_smoke import (  # noqa: E402
    VAEContractError,
    expected_latent_shape,
    validate_vae_config,
)


class GWP05VAEContractTests(unittest.TestCase):
    def test_accepts_pinned_wan_vae_config(self) -> None:
        validate_vae_config(
            {
                "_class_name": "AutoencoderKLWan",
                "z_dim": 48,
                "scale_factor_spatial": 16,
                "scale_factor_temporal": 4,
                "patch_size": 2,
                "latents_mean": [0.0] * 48,
                "latents_std": [1.0] * 48,
            }
        )

    def test_rejects_wrong_latent_dimension(self) -> None:
        with self.assertRaisesRegex(VAEContractError, "z_dim"):
            validate_vae_config(
                {
                    "_class_name": "AutoencoderKLWan",
                    "z_dim": 16,
                    "scale_factor_spatial": 16,
                    "scale_factor_temporal": 4,
                    "patch_size": 2,
                    "latents_mean": [0.0] * 48,
                    "latents_std": [1.0] * 48,
                }
            )

    def test_one_and_five_frames_match_transformer_contract(self) -> None:
        self.assertEqual(expected_latent_shape(1), (1, 48, 1, 24, 20))
        self.assertEqual(expected_latent_shape(5), (1, 48, 2, 24, 20))


if __name__ == "__main__":
    unittest.main()
