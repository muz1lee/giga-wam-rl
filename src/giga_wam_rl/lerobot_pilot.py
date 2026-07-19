import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib

from giga_wam_rl.raw_hdf5_probe import (
    _decode_jpeg,
    _find_registered_asset,
    shifted_action_indices,
)
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _features(action_names: list[str]) -> dict[str, dict[str, Any]]:
    vector_feature = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": action_names,
    }
    features = {
        "observation.state": dict(vector_feature),
        "action": dict(vector_feature),
    }
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        }
    return features


def _first_video_frame(path: Path) -> Any:
    import av

    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        return frame.to_ndarray(format="rgb24")


def _find_episode_video(
    output_root: Path, *, camera_key: str, output_episode_id: int
) -> Path:
    episode_token = f"episode_{output_episode_id:06d}"
    matches = [
        path
        for path in output_root.rglob("*.mp4")
        if camera_key in str(path) and episode_token in path.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one video for {camera_key} {episode_token}, found {len(matches)}"
        )
    return matches[0]


def _validate_parquet(
    parquet_path: Path,
    *,
    expected_state: Any,
    expected_action: Any,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path, columns=["observation.state", "action"]
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != expected_state.shape or action.shape != expected_action.shape:
        raise ValueError(
            f"converted parquet shapes differ: state={state.shape}, action={action.shape}"
        )
    state_error = float(np.max(np.abs(state - expected_state)))
    action_error = float(np.max(np.abs(action - expected_action)))
    if state_error != 0.0 or action_error != 0.0:
        raise ValueError(
            f"converted vectors differ: state_error={state_error}, "
            f"action_error={action_error}"
        )
    return {
        "parquet": str(parquet_path.relative_to(parquet_path.parents[2])),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "state_max_abs_error": state_error,
        "action_max_abs_error": action_error,
        "last_action_repeats_final_target": bool(
            np.array_equal(action[-1], expected_state[-1])
        ),
    }


def run_conversion(
    config_path: Path,
    registry_path: Path,
    *,
    output: TextIO,
) -> int:
    import h5py
    import lerobot
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    registry = load_registry(registry_path)
    validate_registry(registry)

    dataset_config = config["dataset"]
    alignment = config["alignment"]
    camera_map = config["cameras"]["map"]
    probe_config = config["probe"]
    conversion = config["conversion"]

    asset = _find_registered_asset(registry, dataset_config["asset_name"])
    source_root = Path(dataset_config["source_root"]).resolve(strict=True)
    if source_root != Path(asset["path"]).resolve(strict=True):
        raise ValueError("configured source differs from the registered read-only asset")

    workspace_config = registry["workspace"]
    artifact_root = Path(workspace_config["artifact_root"])
    protected_roots = tuple(
        Path(path) for path in workspace_config["protected_roots"]
    )
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
        raise FileExistsError(f"refusing to overwrite conversion manifest: {manifest_path}")

    action_names = list(alignment["action_names"])
    if len(action_names) != int(alignment["physical_action_dimensions"]):
        raise ValueError("action_names do not match physical_action_dimensions")
    fps = int(dataset_config["initial_target_fps"])
    if not conversion["use_videos"]:
        raise ValueError("the GWP pilot contract requires video-backed LeRobot features")

    dataset = LeRobotDataset.create(
        repo_id=conversion["repo_id"],
        root=output_root,
        fps=fps,
        robot_type=conversion["robot_type"],
        features=_features(action_names),
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=4,
        video_backend=conversion["video_backend"],
    )

    pilot_ids = [int(episode_id) for episode_id in probe_config["episode_ids"]]
    source_records = []
    expected_vectors = []
    for output_episode_id, source_episode_id in enumerate(pilot_ids):
        source_path = source_root / f"episode{source_episode_id}.hdf5"
        with h5py.File(source_path, "r") as episode_file:
            states = np.asarray(
                episode_file[alignment["state_key"]], dtype=np.float32
            )
            action_indices = shifted_action_indices(int(states.shape[0]))
            actions = states[action_indices]
            for frame_index in range(states.shape[0]):
                frame = {
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                }
                for model_camera, source_key in camera_map.items():
                    frame[f"observation.images.{model_camera}"] = _decode_jpeg(
                        episode_file[source_key][frame_index]
                    )
                dataset.add_frame(
                    frame, task=dataset_config["canonical_instruction"]
                )
            dataset.save_episode()
        expected_vectors.append((states, actions))
        source_records.append(
            {
                "source_episode_id": source_episode_id,
                "output_episode_id": output_episode_id,
                "source_file": source_path.name,
                "num_frames": int(states.shape[0]),
            }
        )

    parquet_files = sorted(output_root.rglob("*.parquet"))
    video_files = sorted(output_root.rglob("*.mp4"))
    if len(parquet_files) != len(pilot_ids):
        raise ValueError(f"expected {len(pilot_ids)} parquet files")
    if len(video_files) != len(pilot_ids) * len(camera_map):
        raise ValueError(f"expected {len(pilot_ids) * len(camera_map)} videos")

    validations = []
    for output_episode_id, source_episode_id in enumerate(pilot_ids):
        states, actions = expected_vectors[output_episode_id]
        vector_validation = _validate_parquet(
            parquet_files[output_episode_id],
            expected_state=states,
            expected_action=actions,
        )
        camera_validation = {}
        source_path = source_root / f"episode{source_episode_id}.hdf5"
        with h5py.File(source_path, "r") as episode_file:
            for model_camera, source_key in camera_map.items():
                video_path = _find_episode_video(
                    output_root,
                    camera_key=f"observation.images.{model_camera}",
                    output_episode_id=output_episode_id,
                )
                converted_rgb = _first_video_frame(video_path).astype(np.float32)
                source_rgb = _decode_jpeg(episode_file[source_key][0]).astype(
                    np.float32
                )
                direct_mae = float(np.mean(np.abs(converted_rgb - source_rgb)))
                reversed_mae = float(
                    np.mean(np.abs(converted_rgb - source_rgb[..., ::-1]))
                )
                if direct_mae >= reversed_mae:
                    raise ValueError(
                        f"color validation failed for episode {source_episode_id} "
                        f"camera {model_camera}: direct={direct_mae}, "
                        f"reversed={reversed_mae}"
                    )
                camera_validation[model_camera] = {
                    "video": str(video_path.relative_to(output_root)),
                    "first_frame_shape": list(converted_rgb.shape),
                    "source_rgb_mae": direct_mae,
                    "source_channel_reversed_mae": reversed_mae,
                }
        validations.append(
            {
                "source_episode_id": source_episode_id,
                "output_episode_id": output_episode_id,
                "vectors": vector_validation,
                "cameras": camera_validation,
            }
        )

    output_files = []
    for path in sorted(path for path in output_root.rglob("*") if path.is_file()):
        output_files.append(
            {
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "asset_name": dataset_config["asset_name"],
            "root": str(source_root),
            "read_only": True,
            "episodes": source_records,
        },
        "conversion": {
            "format": "LeRobot v3",
            "lerobot_version": getattr(lerobot, "__version__", "unknown"),
            "repo_id": conversion["repo_id"],
            "robot_type": conversion["robot_type"],
            "fps": fps,
            "output_root": str(output_root),
            "action_row_semantics": "action[t] = raw_joint_target[min(t + 1, T - 1)]",
            "decoded_color_contract": config["cameras"][
                "decoded_color_contract"
            ],
            "files": output_files,
        },
        "validation": validations,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"lerobot_pilot status=ok episodes={len(pilot_ids)} "
        f"frames={sum(record['num_frames'] for record in source_records)} "
        f"root={output_root} manifest={manifest_path}",
        file=output,
    )
    return 0


def main(argv: Sequence[str] | None = None, *, output: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert the read-only RoboTwin pilot to LeRobot v3"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if output is None:
        output = sys.stdout
    return run_conversion(arguments.config, arguments.registry, output=output)


if __name__ == "__main__":
    raise SystemExit(main())
