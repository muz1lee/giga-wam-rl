import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.future_flow import (  # noqa: E402
    build_clean_action_timestep,
    future_flow_batch,
)


class FutureFlowTests(unittest.TestCase):
    def test_only_future_latent_is_noised(self) -> None:
        visual_latents = torch.tensor([[[[[2.0]], [[4.0]]]]])
        noise = torch.tensor([[[[[10.0]]]]])
        sigma = torch.tensor([[[[[0.25]]]]])

        future, noisy_future, target = future_flow_batch(
            visual_latents, sigma=sigma, noise=noise
        )

        torch.testing.assert_close(future, torch.tensor([[[[[4.0]]]]]))
        torch.testing.assert_close(noisy_future, torch.tensor([[[[[5.5]]]]]))
        torch.testing.assert_close(target, torch.tensor([[[[[6.0]]]]]))

    def test_clean_action_timestep_only_marks_future_tokens(self) -> None:
        timestep = build_clean_action_timestep(
            torch.tensor([250, 750]),
            num_state_tokens=1,
            num_ref_tokens=3,
            num_action_tokens=2,
            num_future_tokens=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(timestep.shape), (2, 10))
        torch.testing.assert_close(timestep[:, :6], torch.zeros(2, 6))
        torch.testing.assert_close(
            timestep[:, 6:],
            torch.tensor([[250.0] * 4, [750.0] * 4]),
        )

    def test_future_flow_rejects_single_latent_frame(self) -> None:
        with self.assertRaises(ValueError):
            future_flow_batch(
                torch.zeros(1, 48, 1, 2, 2),
                sigma=torch.zeros(1, 1, 1, 1, 1),
                noise=torch.zeros(1, 48, 0, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
