import json

import numpy as np
import pytest
from PIL import Image

from giga_wam_rl.robotwin_collection import (
    DELTA_MASK,
    EpisodeBuffer,
    PolicyActionNormalizer,
    ReplanProposal,
    extract_robotwin_observation,
    fixed_cadence_targets,
    write_episode,
)


def _observation(value: int = 0) -> dict:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image + 1},
            "right_camera": {"rgb": image + 2},
        },
        "joint_action": {
            "vector": np.linspace(-0.5, 0.5, 14, dtype=np.float32) + value,
        },
    }


def _stats() -> dict:
    return {
        "metadata": {"model_dim": 16, "action_horizon": 48},
        "norm_stats": {
            "observation.state": {
                "q01": [-2.0] * 16,
                "q99": [2.0] * 16,
            },
            "action": {
                "q01": [-1.0] * 16,
                "q99": [1.0] * 16,
            },
        },
    }


def test_extract_robotwin_observation_copies_validated_payload() -> None:
    source = _observation()
    payload = extract_robotwin_observation(source)

    assert payload.state.shape == (14,)
    assert payload.cameras["cam_high"].shape == (12, 16, 3)
    assert payload.cameras["cam_left_wrist"].dtype == np.uint8

    source["observation"]["head_camera"]["rgb"][:] = 255
    assert payload.cameras["cam_high"].max() == 0


def test_extract_robotwin_observation_rejects_non_rgb_camera() -> None:
    source = _observation()
    source["observation"]["head_camera"]["rgb"] = np.zeros((12, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="cam_high"):
        extract_robotwin_observation(source)


def test_policy_action_normalizer_uses_delta_arms_and_absolute_grippers() -> None:
    normalizer = PolicyActionNormalizer.from_payload(_stats())
    state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    normalized_state = normalizer.normalize_state(state)

    assert normalized_state.shape == (1, 16)
    assert np.array_equal(normalized_state[0, 14:], np.zeros(2))

    normalized_action = np.zeros((1, 48, 16), dtype=np.float32)
    physical = normalizer.denormalize_action(normalized_action, state)

    assert physical.shape == (48, 14)
    assert np.allclose(physical[:, DELTA_MASK], state[DELTA_MASK])
    assert np.allclose(physical[:, ~DELTA_MASK], 0.0)


def test_policy_action_normalizer_clips_model_values_before_denormalizing() -> None:
    normalizer = PolicyActionNormalizer.from_payload(_stats())
    normalized_action = np.full((1, 48, 16), 4.0, dtype=np.float32)
    state = np.zeros(14, dtype=np.float32)

    physical = normalizer.denormalize_action(
        normalized_action, state, clip_normalized=True
    )

    assert np.allclose(physical, 1.0)


def test_fixed_cadence_targets_interpolate_exactly_to_endpoint() -> None:
    current = np.zeros(14, dtype=np.float32)
    target = np.arange(14, dtype=np.float32)

    path = fixed_cadence_targets(current, target, simulator_steps=15)

    assert path.shape == (15, 14)
    assert np.allclose(path[0], target / 15)
    assert np.array_equal(path[-1], target)


def test_episode_buffer_preserves_causal_observation_action_contract(tmp_path) -> None:
    buffer = EpisodeBuffer(
        task="place_bread_basket",
        instruction="Pick up the bread and place it in the basket.",
        env_seed=7,
        policy_seed=17,
        simulator_hz=250,
        simulator_steps_per_action=15,
    )
    buffer.append_initial(_observation(0), simulator_step=0, wall_time_s=0.0)
    normalized = np.zeros((48, 16), dtype=np.float32)
    physical = np.zeros((48, 14), dtype=np.float32)
    buffer.record_replan(
        ReplanProposal(
            observation_index=0,
            replan_index=0,
            policy_seed=17,
            normalized_action=normalized,
            physical_action=physical,
            committed_length=1,
            inference_time_s=0.25,
        )
    )
    next_observation = _observation(0)
    next_observation["joint_action"]["vector"] = physical[0]
    buffer.append_transition(
        executed_action=physical[0],
        next_observation=next_observation,
        simulator_step=15,
        wall_time_s=0.4,
        replan_index=0,
        proposal_offset=0,
    )

    episode_dir = tmp_path / "episode_000007"
    manifest = write_episode(
        buffer,
        episode_dir,
        success=False,
        termination_reason="smoke_limit",
        code_revision="deadbeef",
        upstream_revisions={"giga_world_policy": "gwp", "robotwin": "rt"},
    )

    assert manifest["num_observations"] == 2
    assert manifest["num_executed_actions"] == 1
    assert manifest["num_replans"] == 1
    assert json.loads((episode_dir / "metadata.json").read_text()) == manifest

    with np.load(episode_dir / "trajectory.npz", allow_pickle=False) as payload:
        assert payload["observation_state"].shape == (2, 14)
        assert payload["executed_action"].shape == (1, 14)
        assert payload["replan_normalized_action"].shape == (1, 48, 16)
        assert payload["replan_physical_action"].shape == (1, 48, 14)
        encoded = payload["cam_high_jpeg"][0].tobytes()
        decoded = np.asarray(Image.open(__import__("io").BytesIO(encoded)))
        assert decoded.shape == (12, 16, 3)


def test_episode_writer_refuses_overwrite(tmp_path) -> None:
    buffer = EpisodeBuffer(
        task="place_bread_basket",
        instruction="instruction",
        env_seed=1,
        policy_seed=2,
        simulator_hz=250,
        simulator_steps_per_action=15,
    )
    buffer.append_initial(_observation(), simulator_step=0, wall_time_s=0.0)
    episode_dir = tmp_path / "episode_000001"
    write_episode(
        buffer,
        episode_dir,
        success=False,
        termination_reason="empty_smoke",
        code_revision="deadbeef",
        upstream_revisions={},
    )

    with pytest.raises(FileExistsError):
        write_episode(
            buffer,
            episode_dir,
            success=False,
            termination_reason="empty_smoke",
            code_revision="deadbeef",
            upstream_revisions={},
        )
