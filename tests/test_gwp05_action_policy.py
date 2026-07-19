import json

import numpy as np
import pytest
import torch

from giga_wam_rl.gwp05_action_policy import (
    GWP05ActionPolicy,
    _load_action_pipeline_options,
)
from giga_wam_rl.robotwin_collection import PolicyActionNormalizer


def _normalizer() -> PolicyActionNormalizer:
    return PolicyActionNormalizer.from_payload(
        {
            "metadata": {"model_dim": 16, "action_horizon": 48},
            "norm_stats": {
                "observation.state": {
                    "q01": [-2.0] * 16,
                    "q99": [2.0] * 16,
                },
                "action": {"q01": [-1.0] * 16, "q99": [1.0] * 16},
            },
        }
    )


class _FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros((1, 48, 16), dtype=torch.bfloat16)


def test_action_policy_uses_16d_checkpoint_contract_and_t_layout() -> None:
    pipeline = _FakePipeline()
    policy = GWP05ActionPolicy(
        pipeline=pipeline,
        normalizer=_normalizer(),
        prompt_embedding=torch.zeros((1, 64, 4096), dtype=torch.bfloat16),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        num_inference_steps=10,
        clip_normalized_actions=True,
    )
    state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    cameras = {
        "cam_high": np.zeros((240, 320, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((240, 320, 3), dtype=np.uint8),
        "cam_right_wrist": np.full((240, 320, 3), 2, dtype=np.uint8),
    }

    prediction = policy.predict(cameras=cameras, state=state, seed=123)

    assert prediction.normalized_action.shape == (48, 16)
    assert prediction.physical_action.shape == (48, 14)
    assert prediction.seed == 123
    assert prediction.inference_time_s >= 0
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["action_dim"] == 16
    assert call["action_chunk"] == 48
    assert call["height"] == 384
    assert call["width"] == 320
    assert call["num_frames"] == 5
    assert call["state"].shape == (1, 16)
    assert call["image"].size == (320, 384)
    assert call["prompt_embeds"].shape == (1, 64, 4096)


def test_action_policy_reseeds_each_replan_request() -> None:
    pipeline = _FakePipeline()
    policy = GWP05ActionPolicy(
        pipeline=pipeline,
        normalizer=_normalizer(),
        prompt_embedding=torch.zeros((1, 64, 4096), dtype=torch.bfloat16),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        num_inference_steps=10,
        clip_normalized_actions=True,
    )
    cameras = {
        name: np.zeros((16, 16, 3), dtype=np.uint8)
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }

    policy.predict(cameras=cameras, state=np.zeros(14), seed=4)
    policy.predict(cameras=cameras, state=np.zeros(14), seed=9)

    assert pipeline.calls[0]["generator"].initial_seed() == 4
    assert pipeline.calls[1]["generator"].initial_seed() == 9


def test_action_pipeline_copies_expand_timesteps_from_base_model(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"expand_timesteps": True, "boundary_ratio": None}),
        encoding="utf-8",
    )

    assert _load_action_pipeline_options(tmp_path) == {
        "expand_timesteps": True,
        "boundary_ratio": None,
    }


def test_action_pipeline_rejects_non_action_only_base_model(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"expand_timesteps": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expand_timesteps=true"):
        _load_action_pipeline_options(tmp_path)
