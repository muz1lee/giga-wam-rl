from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image


PHYSICAL_ACTION_DIM = 14
MODEL_ACTION_DIM = 16
ACTION_HORIZON = 48
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DELTA_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)


@dataclass(frozen=True)
class RobotWinObservation:
    cameras: dict[str, np.ndarray]
    state: np.ndarray


@dataclass(frozen=True)
class ReplanProposal:
    observation_index: int
    replan_index: int
    policy_seed: int
    normalized_action: np.ndarray
    physical_action: np.ndarray
    committed_length: int
    inference_time_s: float

    def __post_init__(self) -> None:
        normalized = np.asarray(self.normalized_action, dtype=np.float32)
        physical = np.asarray(self.physical_action, dtype=np.float32)
        if normalized.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
            raise ValueError(
                "normalized action must have shape "
                f"{(ACTION_HORIZON, MODEL_ACTION_DIM)}, got {normalized.shape}"
            )
        if physical.shape != (ACTION_HORIZON, PHYSICAL_ACTION_DIM):
            raise ValueError(
                "physical action must have shape "
                f"{(ACTION_HORIZON, PHYSICAL_ACTION_DIM)}, got {physical.shape}"
            )
        if not 1 <= int(self.committed_length) <= ACTION_HORIZON:
            raise ValueError("committed length must be in [1, 48]")
        if not np.isfinite(normalized).all() or not np.isfinite(physical).all():
            raise ValueError("replan actions must be finite")
        if self.observation_index < 0 or self.replan_index < 0:
            raise ValueError("replan indices must be non-negative")
        if self.inference_time_s < 0 or not np.isfinite(self.inference_time_s):
            raise ValueError("inference time must be finite and non-negative")
        object.__setattr__(self, "normalized_action", normalized.copy())
        object.__setattr__(self, "physical_action", physical.copy())


def _validated_rgb(name: str, value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {image.dtype}")
    return image.copy()


def extract_robotwin_observation(observation: Mapping[str, Any]) -> RobotWinObservation:
    if not isinstance(observation, Mapping):
        raise TypeError("RoboTwin observation must be a mapping")
    try:
        observation_root = observation["observation"]
        joint_action = observation["joint_action"]
        cameras = {
            "cam_high": _validated_rgb(
                "cam_high", observation_root["head_camera"]["rgb"]
            ),
            "cam_left_wrist": _validated_rgb(
                "cam_left_wrist", observation_root["left_camera"]["rgb"]
            ),
            "cam_right_wrist": _validated_rgb(
                "cam_right_wrist", observation_root["right_camera"]["rgb"]
            ),
        }
        state = np.asarray(joint_action["vector"], dtype=np.float32)
    except KeyError as error:
        raise KeyError(f"RoboTwin observation is missing {error}") from error
    if state.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError(
            f"RoboTwin state must have shape {(PHYSICAL_ACTION_DIM,)}, got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("RoboTwin state must be finite")
    return RobotWinObservation(cameras=cameras, state=state.copy())


def _quantiles(
    stats: Mapping[str, Any], *, minimum_dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    high = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    if min(low.size, high.size) < minimum_dimensions:
        raise ValueError("normalization stats do not cover required dimensions")
    if np.any(high[:minimum_dimensions] <= low[:minimum_dimensions]):
        raise ValueError("normalization q99 must exceed q01")
    return low, high


@dataclass(frozen=True)
class PolicyActionNormalizer:
    state_low: np.ndarray
    state_high: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PolicyActionNormalizer:
        metadata = payload.get("metadata", {})
        if metadata.get("model_dim") != MODEL_ACTION_DIM:
            raise ValueError("norm stats must declare model_dim=16")
        if metadata.get("action_horizon") != ACTION_HORIZON:
            raise ValueError("norm stats must declare action_horizon=48")
        stats = payload["norm_stats"]
        state_low, state_high = _quantiles(
            stats["observation.state"], minimum_dimensions=PHYSICAL_ACTION_DIM
        )
        action_low, action_high = _quantiles(
            stats["action"], minimum_dimensions=PHYSICAL_ACTION_DIM
        )
        return cls(
            state_low=state_low[:PHYSICAL_ACTION_DIM].copy(),
            state_high=state_high[:PHYSICAL_ACTION_DIM].copy(),
            action_low=action_low[:PHYSICAL_ACTION_DIM].copy(),
            action_high=action_high[:PHYSICAL_ACTION_DIM].copy(),
        )

    @classmethod
    def from_json(cls, path: Path) -> PolicyActionNormalizer:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (PHYSICAL_ACTION_DIM,):
            raise ValueError("state must have shape [14]")
        normalized = np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)
        normalized[0, :PHYSICAL_ACTION_DIM] = (state - self.state_low) / (
            self.state_high - self.state_low
        ) * 2.0 - 1.0
        return normalized

    def denormalize_action(
        self,
        normalized_action: np.ndarray,
        state: np.ndarray,
        *,
        clip_normalized: bool = True,
    ) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape == (1, ACTION_HORIZON, MODEL_ACTION_DIM):
            action = action[0]
        if action.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
            raise ValueError("normalized action must have shape [48,16] or [1,48,16]")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (PHYSICAL_ACTION_DIM,):
            raise ValueError("state must have shape [14]")
        model_action = action[:, :PHYSICAL_ACTION_DIM].copy()
        if clip_normalized:
            np.clip(model_action, -1.0, 1.0, out=model_action)
        mixed = (model_action + 1.0) / 2.0 * (
            self.action_high - self.action_low
        ) + self.action_low
        mixed[:, DELTA_MASK] += state[DELTA_MASK]
        return mixed.astype(np.float32, copy=False)


def fixed_cadence_targets(
    current: np.ndarray, target: np.ndarray, *, simulator_steps: int
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if current.shape != (PHYSICAL_ACTION_DIM,) or target.shape != (
        PHYSICAL_ACTION_DIM,
    ):
        raise ValueError("current and target must have shape [14]")
    if simulator_steps <= 0:
        raise ValueError("simulator steps must be positive")
    fractions = np.arange(1, simulator_steps + 1, dtype=np.float32) / simulator_steps
    return current[None] + fractions[:, None] * (target - current)[None]


@dataclass
class EpisodeBuffer:
    task: str
    instruction: str
    env_seed: int
    policy_seed: int
    simulator_hz: int
    simulator_steps_per_action: int
    observations: list[RobotWinObservation] = field(default_factory=list)
    simulator_steps: list[int] = field(default_factory=list)
    wall_times_s: list[float] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    executed_replan_indices: list[int] = field(default_factory=list)
    executed_proposal_offsets: list[int] = field(default_factory=list)
    replans: list[ReplanProposal] = field(default_factory=list)

    def append_initial(
        self,
        observation: Mapping[str, Any],
        *,
        simulator_step: int,
        wall_time_s: float,
    ) -> None:
        if self.observations:
            raise RuntimeError("initial observation already exists")
        self._append_observation(
            observation, simulator_step=simulator_step, wall_time_s=wall_time_s
        )

    def record_replan(self, proposal: ReplanProposal) -> None:
        if not self.observations:
            raise RuntimeError("cannot record replan before initial observation")
        if proposal.observation_index != len(self.observations) - 1:
            raise ValueError(
                "replan observation index does not match current observation"
            )
        if proposal.replan_index != len(self.replans):
            raise ValueError("replan indices must be contiguous")
        self.replans.append(proposal)

    def append_transition(
        self,
        *,
        executed_action: np.ndarray,
        next_observation: Mapping[str, Any],
        simulator_step: int,
        wall_time_s: float,
        replan_index: int,
        proposal_offset: int,
    ) -> None:
        if not self.observations:
            raise RuntimeError("cannot append transition before initial observation")
        action = np.asarray(executed_action, dtype=np.float32)
        if action.shape != (PHYSICAL_ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("executed action must be a finite [14] vector")
        if not 0 <= replan_index < len(self.replans):
            raise ValueError("transition references an unknown replan")
        proposal = self.replans[replan_index]
        if not 0 <= proposal_offset < proposal.committed_length:
            raise ValueError("proposal offset is outside the committed prefix")
        parsed = extract_robotwin_observation(next_observation)
        if not np.allclose(parsed.state, action, rtol=0.0, atol=1e-4):
            raise ValueError(
                "next observation drive target does not match executed action"
            )
        self.executed_actions.append(action.copy())
        self.executed_replan_indices.append(int(replan_index))
        self.executed_proposal_offsets.append(int(proposal_offset))
        self._append_parsed_observation(
            parsed, simulator_step=simulator_step, wall_time_s=wall_time_s
        )

    def _append_observation(
        self,
        observation: Mapping[str, Any],
        *,
        simulator_step: int,
        wall_time_s: float,
    ) -> None:
        self._append_parsed_observation(
            extract_robotwin_observation(observation),
            simulator_step=simulator_step,
            wall_time_s=wall_time_s,
        )

    def _append_parsed_observation(
        self,
        parsed: RobotWinObservation,
        *,
        simulator_step: int,
        wall_time_s: float,
    ) -> None:
        if self.simulator_steps and simulator_step <= self.simulator_steps[-1]:
            raise ValueError("simulator steps must be strictly increasing")
        if self.wall_times_s and wall_time_s < self.wall_times_s[-1]:
            raise ValueError("wall times must be monotonic")
        if self.observations:
            expected_shapes = {
                name: self.observations[0].cameras[name].shape for name in CAMERA_NAMES
            }
            actual_shapes = {name: parsed.cameras[name].shape for name in CAMERA_NAMES}
            if actual_shapes != expected_shapes:
                raise ValueError("camera shapes changed within the episode")
        self.observations.append(parsed)
        self.simulator_steps.append(int(simulator_step))
        self.wall_times_s.append(float(wall_time_s))


def _encode_jpeg(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=90, subsampling=0)
    return output.getvalue()


def _fixed_bytes(values: list[bytes]) -> np.ndarray:
    max_size = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"S{max_size}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_arrays(buffer: EpisodeBuffer) -> dict[str, np.ndarray]:
    if not buffer.observations:
        raise ValueError("episode has no observations")
    if len(buffer.observations) != len(buffer.executed_actions) + 1:
        raise ValueError("episode must have exactly one more observation than action")
    replans = buffer.replans
    arrays: dict[str, np.ndarray] = {
        "observation_state": np.stack(
            [observation.state for observation in buffer.observations]
        ).astype(np.float32),
        "executed_action": np.asarray(
            buffer.executed_actions, dtype=np.float32
        ).reshape(-1, PHYSICAL_ACTION_DIM),
        "observation_simulator_step": np.asarray(
            buffer.simulator_steps, dtype=np.int64
        ),
        "observation_wall_time_s": np.asarray(buffer.wall_times_s, dtype=np.float64),
        "executed_replan_index": np.asarray(
            buffer.executed_replan_indices, dtype=np.int32
        ),
        "executed_proposal_offset": np.asarray(
            buffer.executed_proposal_offsets, dtype=np.int32
        ),
        "replan_observation_index": np.asarray(
            [proposal.observation_index for proposal in replans], dtype=np.int64
        ),
        "replan_policy_seed": np.asarray(
            [proposal.policy_seed for proposal in replans], dtype=np.int64
        ),
        "replan_committed_length": np.asarray(
            [proposal.committed_length for proposal in replans], dtype=np.int32
        ),
        "replan_inference_time_s": np.asarray(
            [proposal.inference_time_s for proposal in replans], dtype=np.float64
        ),
        "replan_normalized_action": np.asarray(
            [proposal.normalized_action for proposal in replans], dtype=np.float32
        ).reshape(-1, ACTION_HORIZON, MODEL_ACTION_DIM),
        "replan_physical_action": np.asarray(
            [proposal.physical_action for proposal in replans], dtype=np.float32
        ).reshape(-1, ACTION_HORIZON, PHYSICAL_ACTION_DIM),
    }
    for camera_name in CAMERA_NAMES:
        arrays[f"{camera_name}_jpeg"] = _fixed_bytes(
            [
                _encode_jpeg(observation.cameras[camera_name])
                for observation in buffer.observations
            ]
        )
    return arrays


def write_episode(
    buffer: EpisodeBuffer,
    episode_dir: Path,
    *,
    success: bool,
    termination_reason: str,
    code_revision: str,
    upstream_revisions: Mapping[str, str],
) -> dict[str, Any]:
    episode_dir = episode_dir.resolve(strict=False)
    if episode_dir.exists():
        raise FileExistsError(f"refusing to overwrite episode: {episode_dir}")
    episode_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{episode_dir.name}.tmp-", dir=episode_dir.parent)
    )
    try:
        trajectory_path = temporary_dir / "trajectory.npz"
        arrays = _episode_arrays(buffer)
        with trajectory_path.open("wb") as output_file:
            np.savez(output_file, **arrays)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "task": buffer.task,
            "instruction": buffer.instruction,
            "env_seed": int(buffer.env_seed),
            "policy_seed": int(buffer.policy_seed),
            "success": bool(success),
            "termination_reason": str(termination_reason),
            "simulator_hz": int(buffer.simulator_hz),
            "simulator_steps_per_action": int(buffer.simulator_steps_per_action),
            "num_observations": len(buffer.observations),
            "num_executed_actions": len(buffer.executed_actions),
            "num_replans": len(buffer.replans),
            "code_revision": str(code_revision),
            "upstream_revisions": dict(upstream_revisions),
            "trajectory_file": "trajectory.npz",
            "trajectory_sha256": _sha256_file(trajectory_path),
            "image_encoding": {
                "format": "jpeg",
                "quality": 90,
                "color": "rgb",
                "cameras": list(CAMERA_NAMES),
            },
        }
        (temporary_dir / "metadata.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(episode_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest
