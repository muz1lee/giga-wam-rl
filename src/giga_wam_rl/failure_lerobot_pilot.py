import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib

from giga_wam_rl.failure_rollout_probe import (
    CAMERA_KEYS,
    _is_within,
    _registered_read_only_roots,
    recover_failure_episode,
)
from giga_wam_rl.lerobot_pilot import (
    _features,
    _find_camera_video,
    _validate_parquet,
    _video_frame,
)
from giga_wam_rl.raw_hdf5_probe import candidate_window_count
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


ACTION_NAMES = [
    "fl_joint1",
    "fl_joint2",
    "fl_joint3",
    "fl_joint4",
    "fl_joint5",
    "fl_joint6",
    "left_gripper",
    "fr_joint1",
    "fr_joint2",
    "fr_joint3",
    "fr_joint4",
    "fr_joint5",
    "fr_joint6",
    "right_gripper",
]


def _camera_frame_mae(converted: Any, source: Any, *, maximum_mae: float) -> float:
    import numpy as np

    converted_array = np.asarray(converted)
    source_array = np.asarray(source)
    if converted_array.shape != source_array.shape:
        raise ValueError(
            f"converted/source image shapes differ: "
            f"{converted_array.shape} != {source_array.shape}"
        )
    error = float(
        np.mean(
            np.abs(converted_array.astype(np.float32) - source_array.astype(np.float32))
        )
    )
    if error > maximum_mae:
        raise ValueError(f"converted camera frame MAE {error} exceeds {maximum_mae}")
    return error


def _registered_asset(registry: dict[str, Any], asset_name: str) -> dict[str, Any]:
    matches = [asset for asset in registry["assets"] if asset.get("name") == asset_name]
    if len(matches) != 1:
        raise ValueError(f"expected one registered asset {asset_name!r}")
    asset = matches[0]
    if asset.get("owner") != "student" or asset.get("read_only") is not True:
        raise ValueError("failure source must be registered student read-only")
    return asset


def run_conversion(
    config_path: Path,
    registry_path: Path,
    *,
    output: TextIO,
) -> int:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    registry = load_registry(registry_path)
    validate_registry(registry)

    dataset_config = config["dataset"]
    conversion = config["conversion"]
    probe_config = config["probe"]
    asset = _registered_asset(registry, dataset_config["asset_name"])
    asset_root = Path(asset["path"]).resolve(strict=True)
    read_only_roots = _registered_read_only_roots(registry)
    for episode_config in config["episodes"]:
        sidecar = Path(episode_config["sidecar_path"]).resolve(strict=True)
        if not _is_within(sidecar, asset_root):
            raise ValueError("failure sidecar is outside its registered asset")

    workspace_config = registry["workspace"]
    artifact_root = Path(workspace_config["artifact_root"])
    protected_roots = tuple(Path(path) for path in workspace_config["protected_roots"])
    output_root = validate_output_root(
        Path(conversion["output_root"]),
        artifact_root=artifact_root,
        protected_roots=protected_roots,
    )
    manifest_path = validate_output_root(
        Path(conversion["output_manifest"]),
        artifact_root=artifact_root,
        protected_roots=protected_roots,
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite converted dataset: {output_root}")
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite conversion manifest: {manifest_path}"
        )
    if conversion["use_videos"] is not True:
        raise ValueError("the GWP failure pilot requires video-backed features")

    dataset = LeRobotDataset.create(
        repo_id=conversion["repo_id"],
        root=output_root,
        fps=int(conversion["fps"]),
        robot_type=conversion["robot_type"],
        features=_features(ACTION_NAMES),
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=4,
        video_backend=conversion["video_backend"],
    )

    expected_vectors = []
    expected_camera_frames = []
    episode_records = []
    horizon = int(probe_config["action_horizon"])
    for output_episode_id, episode_config in enumerate(config["episodes"]):
        recovered = recover_failure_episode(
            episode_config,
            asset_root=asset_root,
            read_only_roots=read_only_roots,
        )
        rows = recovered["rows"]
        states = np.stack([row["observation.state"] for row in rows])
        actions = np.stack([row["action"] for row in rows])
        for row in rows:
            frame = {
                "observation.state": row["observation.state"],
                "action": row["action"],
                "task": row["task"],
            }
            for camera_key in CAMERA_KEYS:
                frame[camera_key] = row[camera_key]
            dataset.add_frame(frame)
        dataset.save_episode()
        expected_vectors.append((states, actions))
        expected_camera_frames.append(
            {
                camera_key: (rows[0][camera_key], rows[-1][camera_key])
                for camera_key in CAMERA_KEYS
            }
        )
        sidecar_row = recovered["sidecar_row"]
        episode_records.append(
            {
                "output_episode_id": output_episode_id,
                "source_episode_index": recovered["episode_index"],
                "seed": sidecar_row["seed"],
                "prompt": sidecar_row["prompt"],
                "success": False,
                "num_frames": len(rows),
                "candidate_48_step_windows": candidate_window_count(
                    len(rows), horizon=horizon
                ),
                "source_rollout_dir": str(recovered["rollout_dir"]),
            }
        )
    dataset.finalize()

    parquet_files = sorted((output_root / "data").rglob("*.parquet"))
    video_files = sorted((output_root / "videos").rglob("*.mp4"))
    if not parquet_files:
        raise ValueError("converted failure dataset has no data parquet")
    if len(video_files) != len(CAMERA_KEYS):
        raise ValueError(
            f"expected {len(CAMERA_KEYS)} video chunks, found {len(video_files)}"
        )

    validations = []
    output_episode_start = 0
    for output_episode_id, (states, actions) in enumerate(expected_vectors):
        camera_validation = {}
        for camera_key in CAMERA_KEYS:
            video_path = _find_camera_video(output_root, camera_key=camera_key)
            source_first, source_last = expected_camera_frames[output_episode_id][
                camera_key
            ]
            first_mae = _camera_frame_mae(
                _video_frame(video_path, output_episode_start),
                source_first,
                maximum_mae=5.0,
            )
            last_mae = _camera_frame_mae(
                _video_frame(video_path, output_episode_start + len(states) - 1),
                source_last,
                maximum_mae=5.0,
            )
            camera_validation[camera_key] = {
                "video": str(video_path.relative_to(output_root)),
                "first_frame_mae_0_255": first_mae,
                "last_frame_mae_0_255": last_mae,
            }
        validations.append(
            {
                "output_episode_id": output_episode_id,
                "cameras": camera_validation,
                **_validate_parquet(
                    parquet_files,
                    output_root=output_root,
                    output_episode_id=output_episode_id,
                    expected_state=states,
                    expected_action=actions,
                ),
            }
        )
        output_episode_start += len(states)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "asset_name": dataset_config["asset_name"],
            "owner": asset["owner"],
            "read_only": True,
            "lingbot_actions_used": False,
            "lingbot_latents_used": False,
        },
        "output": {
            "root": str(output_root),
            "repo_id": conversion["repo_id"],
            "fps_metadata": int(conversion["fps"]),
            "num_episodes": len(episode_records),
            "num_frames": sum(record["num_frames"] for record in episode_records),
        },
        "alignment": {
            "state": "captured 14D RoboTwin joint drive target",
            "action": "next captured 14D joint drive target",
            "terminal_action": "repeat final drive target",
            "source_capture": "one keyframe per four high-level EE actions",
            "source_timestamps_available": False,
            "cadence_matches_success_demonstrations": False,
            "intended_use": "pipeline and failure-future overfit warm-start only",
        },
        "episodes": episode_records,
        "validations": validations,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), file=output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert sparse archived failure observations to LeRobot v3"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_conversion(args.config, args.registry, output=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
