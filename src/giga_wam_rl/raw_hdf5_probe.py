import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib

from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


def candidate_window_count(num_frames: int, *, horizon: int) -> int:
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return max(num_frames - horizon, 0)


def shifted_action_indices(num_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return [min(index + 1, num_frames - 1) for index in range(num_frames)]


def sample_indices(
    start: int,
    *,
    horizon: int,
    frame_offsets: tuple[int, ...],
) -> dict[str, int | list[int]]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not frame_offsets or frame_offsets[0] != 0:
        raise ValueError("frame_offsets must start at zero")
    if tuple(sorted(frame_offsets)) != frame_offsets:
        raise ValueError("frame_offsets must be sorted")
    if frame_offsets[-1] != horizon:
        raise ValueError("the final frame offset must equal horizon")
    return {
        "state": start,
        "actions": list(range(start + 1, start + horizon + 1)),
        "images": [start + offset for offset in frame_offsets],
    }


def _sha256_array(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _decode_jpeg(encoded: Any) -> Any:
    import cv2
    import numpy as np

    if isinstance(encoded, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(encoded, dtype=np.uint8)
    else:
        buffer = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode a camera JPEG")
    return image


def _find_registered_asset(registry: dict[str, Any], asset_name: str) -> dict[str, Any]:
    matches = [
        asset for asset in registry.get("assets", []) if asset.get("name") == asset_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one registered asset named {asset_name!r}")
    asset = matches[0]
    if asset.get("owner") != "student" or asset.get("read_only") is not True:
        raise ValueError(f"source asset must be registered student read-only: {asset_name}")
    return asset


def _discover_episodes(
    source_root: Path,
    *,
    file_glob: str,
    id_regex: str,
) -> list[tuple[int, Path]]:
    pattern = re.compile(id_regex)
    episodes = []
    for path in source_root.glob(file_glob):
        match = pattern.fullmatch(path.name)
        if match:
            episodes.append((int(match.group(1)), path))
    return sorted(episodes)


def _probe_camera(dataset: Any, image_indices: list[int]) -> dict[str, Any]:
    import numpy as np

    images = np.stack([_decode_jpeg(dataset[index]) for index in image_indices])
    return {
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "minimum": int(images.min()),
        "maximum": int(images.max()),
        "decoded_direct_sha256": _sha256_array(images),
        "channel_reversed_sha256": _sha256_array(images[..., ::-1]),
    }


def run_probe(
    config_path: Path,
    registry_path: Path,
    *,
    output: TextIO,
) -> int:
    import h5py
    import numpy as np

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    registry = load_registry(registry_path)
    validate_registry(registry)

    dataset_config = config["dataset"]
    alignment = config["alignment"]
    camera_config = config["cameras"]
    probe_config = config["probe"]

    asset = _find_registered_asset(registry, dataset_config["asset_name"])
    source_root = Path(dataset_config["source_root"]).resolve(strict=True)
    registered_root = Path(asset["path"]).resolve(strict=True)
    if source_root != registered_root:
        raise ValueError(
            f"configured source differs from registered asset: {source_root} != "
            f"{registered_root}"
        )

    workspace_config = registry["workspace"]
    manifest_path = validate_output_root(
        Path(probe_config["output_manifest"]),
        artifact_root=Path(workspace_config["artifact_root"]),
        protected_roots=tuple(
            Path(path) for path in workspace_config["protected_roots"]
        ),
    )

    horizon = int(alignment["action_horizon"])
    frame_offsets = tuple(int(offset) for offset in alignment["frame_offsets"])
    action_shift = int(alignment["action_start_offset"])
    if action_shift != 1:
        raise ValueError("raw target alignment requires action_start_offset = 1")
    indices = sample_indices(
        int(probe_config["sample_start"]),
        horizon=horizon,
        frame_offsets=frame_offsets,
    )

    episodes = _discover_episodes(
        source_root,
        file_glob=dataset_config["episode_file_glob"],
        id_regex=dataset_config["episode_id_regex"],
    )
    expected_count = int(dataset_config["expected_episode_count"])
    if len(episodes) != expected_count:
        raise ValueError(f"expected {expected_count} episodes, found {len(episodes)}")

    state_key = alignment["state_key"]
    component_keys = alignment["action_component_keys"]
    camera_map = camera_config["map"]
    pilot_ids = set(int(episode_id) for episode_id in probe_config["episode_ids"])
    lengths = []
    episode_inventory = []
    pilot_results = []

    for episode_id, episode_path in episodes:
        stat = episode_path.stat()
        with h5py.File(episode_path, "r") as episode_file:
            vector = episode_file[state_key]
            num_frames = int(vector.shape[0])
            lengths.append(num_frames)
            episode_inventory.append(
                {
                    "episode_id": episode_id,
                    "num_frames": num_frames,
                    "candidate_windows": candidate_window_count(
                        num_frames, horizon=horizon
                    ),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            if episode_id not in pilot_ids:
                continue
            if indices["images"][-1] >= num_frames:
                raise ValueError(f"pilot episode {episode_id} is too short")

            components = []
            component_shapes = {}
            for key in component_keys:
                values = np.asarray(episode_file[key])
                component_shapes[key] = list(values.shape)
                components.append(values if values.ndim == 2 else values[:, None])
            concatenated = np.concatenate(components, axis=1)
            vector_values = np.asarray(vector)
            if concatenated.shape != vector_values.shape:
                raise ValueError(
                    f"episode {episode_id} component/vector shapes differ: "
                    f"{concatenated.shape} != {vector_values.shape}"
                )

            action_indices = indices["actions"]
            state_index = int(indices["state"])
            pilot_results.append(
                {
                    "episode_id": episode_id,
                    "file": episode_path.name,
                    "component_shapes": component_shapes,
                    "vector_shape": list(vector_values.shape),
                    "vector_dtype": str(vector_values.dtype),
                    "component_vector_max_abs_error": float(
                        np.max(np.abs(concatenated - vector_values))
                    ),
                    "sample": {
                        "indices": indices,
                        "state_shape": list(vector_values[state_index].shape),
                        "action_chunk_shape": list(
                            vector_values[action_indices].shape
                        ),
                        "state_sha256": _sha256_array(vector_values[state_index]),
                        "action_chunk_sha256": _sha256_array(
                            vector_values[action_indices]
                        ),
                        "cameras": {
                            model_key: _probe_camera(
                                episode_file[source_key], indices["images"]
                            )
                            for model_key, source_key in camera_map.items()
                        },
                    },
                }
            )

    missing_pilots = pilot_ids - {result["episode_id"] for result in pilot_results}
    if missing_pilots:
        raise ValueError(f"pilot episodes not found: {sorted(missing_pilots)}")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "asset_name": dataset_config["asset_name"],
            "root": str(source_root),
            "owner": asset["owner"],
            "read_only": asset["read_only"],
            "success_only": dataset_config["success_only"],
        },
        "contract": {
            "state_key": state_key,
            "action_component_keys": component_keys,
            "physical_action_dimensions": alignment["physical_action_dimensions"],
            "model_action_dimensions": alignment["model_action_dimensions"],
            "model_padding_dimensions": alignment["model_padding_dimensions"],
            "action_start_offset": action_shift,
            "action_horizon": horizon,
            "frame_offsets": list(frame_offsets),
            "camera_map": camera_map,
            "decoded_color_contract": camera_config["decoded_color_contract"],
            "deduplicate_adjacent_frames": dataset_config[
                "deduplicate_adjacent_frames"
            ],
            "canonical_instruction": dataset_config["canonical_instruction"],
            "source_simulation_hz": dataset_config["source_simulation_hz"],
            "source_save_every_steps": dataset_config["source_save_every_steps"],
            "nominal_source_observation_hz": dataset_config[
                "nominal_source_observation_hz"
            ],
            "initial_target_fps": dataset_config["initial_target_fps"],
        },
        "inventory": {
            "episode_count": len(episodes),
            "total_frames": sum(lengths),
            "minimum_frames": min(lengths),
            "maximum_frames": max(lengths),
            "total_candidate_windows": sum(
                episode["candidate_windows"] for episode in episode_inventory
            ),
            "episodes": episode_inventory,
        },
        "pilot_results": sorted(pilot_results, key=lambda result: result["episode_id"]),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"raw_hdf5_probe status=ok episodes={len(episodes)} "
        f"pilot_episodes={len(pilot_results)} manifest={manifest_path}",
        file=output,
    )
    return 0


def main(argv: Sequence[str] | None = None, *, output: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe raw RoboTwin HDF5 read-only")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if output is None:
        output = sys.stdout
    return run_probe(arguments.config, arguments.registry, output=output)


if __name__ == "__main__":
    raise SystemExit(main())
