import runpy
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FailureTrainingConfigTests(unittest.TestCase):
    def test_overfit_config_uses_clean_16d_valid_window_path(self) -> None:
        namespace = runpy.run_path(
            str(
                PROJECT_ROOT
                / "configs"
                / "experiments"
                / "place_bread_failure_future_overfit.py"
            )
        )
        config = namespace["config"]
        train_loader = config["dataloaders"]["train"]
        dataset = train_loader["data_or_config"][0]
        transform = train_loader["transform"]

        self.assertEqual(
            config["runners"],
            ["giga_wam_rl.failure_future_trainer.FailureFutureTrainerMoT"],
        )
        self.assertEqual(dataset["_class_name"], "ValidWindowWAMLeRobotDataset")
        self.assertEqual(dataset["valid_horizon"], 48)
        self.assertEqual(len(dataset["selected_raw_indices"]), 8)
        self.assertEqual(transform["type"], "DeterministicWALeRobotTransforms")
        self.assertEqual(transform["model_action_dim"], 16)
        self.assertTrue(config["models"]["expand_timesteps"])
        self.assertFalse(config["launch"]["until_completion"])
        self.assertFalse(config["train"]["with_ema"])


if __name__ == "__main__":
    unittest.main()
