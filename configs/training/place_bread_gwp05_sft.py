"""Controlled place-bread post-training config derived from pinned GWP-0.5."""

from copy import deepcopy
import os
from pathlib import Path
import runpy


upstream_root = Path(os.environ["GWP05_UPSTREAM_ROOT"]).resolve(strict=True)
upstream_config_path = (
    upstream_root / "configs" / "giga_world_policy_0_5_agilex_finetune.py"
)
config = deepcopy(runpy.run_path(str(upstream_config_path))["config"])

# The pinned checkpoint has 16D action weights. The official 32D data setting is a
# stale interface and fails against this transformer.
transform = config["dataloaders"]["train"]["transform"]
transform["model_action_dim"] = 16

# A failed research job must return control to us instead of being relaunched forever.
config["launch"]["until_completion"] = False

config["models"]["enable_gradient_checkpointing"] = (
    os.environ.get("GWP05_GRADIENT_CHECKPOINTING", "1") == "1"
)
config["train"]["max_steps"] = int(os.environ.get("GWP05_MAX_STEPS", "2"))
config["train"]["checkpoint_interval"] = int(
    os.environ.get("GWP05_CHECKPOINT_INTERVAL", "2")
)
config["train"]["log_interval"] = int(os.environ.get("GWP05_LOG_INTERVAL", "1"))
