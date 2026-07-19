import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giga_wam_rl.failure_t5_cache import _episode_prompts  # noqa: E402


class FailureT5CacheTests(unittest.TestCase):
    def test_prompts_follow_config_episode_order_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sidecar_path = Path(temporary_directory) / "negative.jsonl"
            sidecar_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "episode_index": 1,
                            "seed": 11,
                            "success": False,
                            "prompt": "second prompt",
                        },
                        {
                            "episode_index": 0,
                            "seed": 10,
                            "success": False,
                            "prompt": "first prompt",
                        },
                    ]
                )
                + "\n"
            )
            config = {
                "episodes": [
                    {
                        "sidecar_path": str(sidecar_path),
                        "episode_index": 0,
                        "seed": 10,
                    },
                    {
                        "sidecar_path": str(sidecar_path),
                        "episode_index": 1,
                        "seed": 11,
                    },
                ]
            }

            prompts = _episode_prompts(config)

        self.assertEqual(prompts, ["first prompt", "second prompt"])


if __name__ == "__main__":
    unittest.main()
