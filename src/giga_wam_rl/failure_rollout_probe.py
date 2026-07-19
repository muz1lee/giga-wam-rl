import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib

from giga_wam_rl.raw_hdf5_probe import candidate_window_count
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
_OBS_CHUNK_PATTERN = re.compile(r"obs_data_(\d+)\.pt")


def parse_obs_chunk_index(path: Path) -> int:
    match = _OBS_CHUNK_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not an obs_data chunk: {path}")
    return int(match.group(1))


def _observation_fingerprint(observation: dict[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key in (*CAMERA_KEYS, STATE_KEY):
        if key not in observation:
            raise ValueError(f"observation is missing {key}")
        value = np.asarray(observation[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    task = observation.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("observation task must be a non-empty string")
    digest.update(task.encode())
    return digest.hexdigest()


def _maximum_overlap(left: list[str], right: list[str]) -> int:
    for overlap in range(min(len(left), len(right)), 0, -1):
        if left[-overlap:] == right[:overlap]:
            return overlap
    return 0


def stitch_observation_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[int]]:
    stitched: list[dict[str, Any]] = []
    stitched_fingerprints: list[str] = []
    overlaps: list[int] = []
    for chunk_number, chunk in enumerate(chunks):
        rows = list(chunk)
        if not rows:
            raise ValueError(f"observation chunk {chunk_number} is empty")
        fingerprints = [_observation_fingerprint(row) for row in rows]
        overlap = (
            _maximum_overlap(stitched_fingerprints, fingerprints) if stitched else 0
        )
        if stitched:
            overlaps.append(overlap)
        stitched.extend(rows[overlap:])
        stitched_fingerprints.extend(fingerprints[overlap:])
    return stitched, overlaps


def causal_action_rows(
    observations: Sequence[dict[str, Any]],
    *,
    drop_terminal_duplicates: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import numpy as np

    unique_terminal = list(observations)
    if not unique_terminal:
        raise ValueError("failure trajectory has no observations")
    terminal_duplicates_dropped = 0
    if drop_terminal_duplicates:
        fingerprints = [_observation_fingerprint(row) for row in unique_terminal]
        while len(unique_terminal) > 1 and fingerprints[-1] == fingerprints[-2]:
            unique_terminal.pop()
            fingerprints.pop()
            terminal_duplicates_dropped += 1

    rows = []
    for index, observation in enumerate(unique_terminal):
        next_index = min(index + 1, len(unique_terminal) - 1)
        next_observation = unique_terminal[next_index]
        row = {key: np.asarray(observation[key]) for key in CAMERA_KEYS}
        row[STATE_KEY] = np.asarray(observation[STATE_KEY], dtype=np.float32)
        row["action"] = np.asarray(next_observation[STATE_KEY], dtype=np.float32)
        row["task"] = observation["task"]
        rows.append(row)
    return rows, {
        "input_observations": len(observations),
        "terminal_duplicates_dropped": terminal_duplicates_dropped,
        "causal_rows": len(rows),
    }


def _load_observation_chunk(path: Path) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    numpy_core = np._core if hasattr(np, "_core") else np.core
    reconstruct = numpy_core.multiarray._reconstruct
    safe_globals = [
        reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.uint8)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        chunk = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(chunk, list) or not all(isinstance(row, dict) for row in chunk):
        raise ValueError(f"{path} must contain a list of observation dictionaries")
    for row in chunk:
        _observation_fingerprint(row)
        if tuple(np.asarray(row[STATE_KEY]).shape) != (14,):
            raise ValueError(f"{path} has a non-14D observation.state")
        for camera_key in CAMERA_KEYS:
            image = np.asarray(row[camera_key])
            if image.dtype != np.uint8 or tuple(image.shape) != (240, 320, 3):
                raise ValueError(
                    f"{path} has invalid {camera_key}: {image.shape} {image.dtype}"
                )
    return chunk


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _registered_read_only_roots(registry: dict[str, Any]) -> list[Path]:
    return [
        Path(asset["path"]).resolve(strict=True)
        for asset in registry.get("assets", [])
        if asset.get("owner") == "student" and asset.get("read_only") is True
    ]


def _select_episode(
    rows: Sequence[dict[str, Any]], *, episode_index: int
) -> dict[str, Any]:
    matches = [
        row for row in rows if int(row.get("episode_index", -1)) == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one sidecar row for episode {episode_index}, found {len(matches)}"
        )
    row = matches[0]
    if row.get("success") is not False:
        raise ValueError(f"episode {episode_index} is not labelled as failure")
    return row


def _observation_files_from_sidecar(
    sidecar_row: dict[str, Any],
    *,
    asset_root: Path,
    read_only_roots: Sequence[Path],
) -> list[Path]:
    files = sidecar_row.get("files")
    if not isinstance(files, list):
        raise ValueError("failure sidecar row has no files list")
    selected = []
    names = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("failure sidecar files entries must be dictionaries")
        name = entry.get("name")
        if not isinstance(name, str) or _OBS_CHUNK_PATTERN.fullmatch(name) is None:
            continue
        if name in names:
            raise ValueError(f"duplicate sidecar observation file: {name}")
        names.add(name)
        archive_path = Path(entry["path"])
        source_path = Path(entry["source"])
        if not archive_path.is_absolute() or not source_path.is_absolute():
            raise ValueError("sidecar tensor paths must be absolute")
        if not _is_within(archive_path, asset_root):
            raise ValueError(
                f"sidecar archive is outside registered asset: {archive_path}"
            )
        archive_target = archive_path.resolve(strict=True)
        source_target = source_path.resolve(strict=True)
        if archive_target != source_target:
            raise ValueError(f"sidecar archive/source targets differ for {name}")
        if not any(_is_within(source_target, root) for root in read_only_roots):
            raise ValueError(
                f"sidecar source is not registered read-only: {source_target}"
            )
        selected.append(archive_path)
    if not selected:
        raise ValueError("sidecar files list has no obs_data chunks")
    return sorted(selected, key=parse_obs_chunk_index)


def recover_failure_episode(
    episode_config: dict[str, Any],
    *,
    asset_root: Path,
    read_only_roots: Sequence[Path],
) -> dict[str, Any]:
    episode_index = int(episode_config["episode_index"])
    sidecar_path = Path(episode_config["sidecar_path"]).resolve(strict=True)
    if not _is_within(sidecar_path, asset_root):
        raise ValueError("failure sidecar is outside its registered asset")
    sidecar_row = _select_episode(
        _read_jsonl(sidecar_path), episode_index=episode_index
    )
    if int(sidecar_row.get("seed", -1)) != int(episode_config["seed"]):
        raise ValueError(f"episode {episode_index} seed differs from config")
    rollout_dir = Path(sidecar_row["server_rollout_dir"]).resolve(strict=True)
    if not any(_is_within(rollout_dir, root) for root in read_only_roots):
        raise ValueError(
            f"rollout directory is not registered read-only: {rollout_dir}"
        )
    chunk_paths = _observation_files_from_sidecar(
        sidecar_row,
        asset_root=asset_root,
        read_only_roots=read_only_roots,
    )
    chunks = [_load_observation_chunk(path) for path in chunk_paths]
    stitched, overlaps = stitch_observation_chunks(chunks)
    rows, causal_summary = causal_action_rows(stitched, drop_terminal_duplicates=True)
    return {
        "episode_index": episode_index,
        "sidecar_path": sidecar_path,
        "sidecar_row": sidecar_row,
        "rollout_dir": rollout_dir,
        "chunk_paths": chunk_paths,
        "chunk_lengths": [len(chunk) for chunk in chunks],
        "overlaps": overlaps,
        "stitched": stitched,
        "rows": rows,
        "causal_summary": causal_summary,
    }


def run_probe(
    config_path: Path,
    registry_path: Path,
    *,
    output: TextIO,
) -> int:
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    registry = load_registry(registry_path)
    validate_registry(registry)

    dataset_config = config["dataset"]
    probe_config = config["probe"]
    asset_name = dataset_config["asset_name"]
    assets = [asset for asset in registry["assets"] if asset.get("name") == asset_name]
    if len(assets) != 1 or assets[0].get("read_only") is not True:
        raise ValueError(f"expected one registered read-only asset {asset_name!r}")
    asset_root = Path(assets[0]["path"]).resolve(strict=True)
    workspace_config = registry["workspace"]
    manifest_path = validate_output_root(
        Path(probe_config["output_manifest"]),
        artifact_root=Path(workspace_config["artifact_root"]),
        protected_roots=tuple(
            Path(path) for path in workspace_config["protected_roots"]
        ),
    )
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite probe manifest: {manifest_path}")

    read_only_roots = _registered_read_only_roots(registry)
    horizon = int(probe_config["action_horizon"])
    episode_results = []
    source_sidecars = set()
    for episode_config in config["episodes"]:
        recovered = recover_failure_episode(
            episode_config,
            asset_root=asset_root,
            read_only_roots=read_only_roots,
        )
        episode_index = recovered["episode_index"]
        sidecar_path = recovered["sidecar_path"]
        source_sidecars.add(str(sidecar_path))
        sidecar_row = recovered["sidecar_row"]
        rollout_dir = recovered["rollout_dir"]
        chunk_paths = recovered["chunk_paths"]
        stitched = recovered["stitched"]
        rows = recovered["rows"]
        episode_results.append(
            {
                "episode_index": episode_index,
                "sidecar": str(sidecar_path),
                "seed": sidecar_row.get("seed"),
                "task": sidecar_row.get("task"),
                "prompt": sidecar_row.get("prompt"),
                "success": False,
                "abort_reason": sidecar_row.get("abort_reason"),
                "rollout_dir": str(rollout_dir),
                "chunk_indices": [parse_obs_chunk_index(path) for path in chunk_paths],
                "chunk_lengths": recovered["chunk_lengths"],
                "chunk_overlaps": recovered["overlaps"],
                "raw_chunk_rows": sum(recovered["chunk_lengths"]),
                "stitched_observations": len(stitched),
                **recovered["causal_summary"],
                "candidate_48_step_windows": candidate_window_count(
                    len(rows), horizon=horizon
                ),
                "first_observation_sha256": _observation_fingerprint(stitched[0]),
                "last_observation_sha256": _observation_fingerprint(stitched[-1]),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "asset_name": asset_name,
            "sidecars": sorted(source_sidecars),
            "owner": assets[0]["owner"],
            "read_only": True,
        },
        "alignment": {
            "state_semantics": "RoboTwin joint drive target",
            "action_semantics": "next captured joint drive target",
            "physical_action_dimensions": 14,
            "action_horizon": horizon,
            "lingbot_internal_actions_used": False,
        },
        "episodes": episode_results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), file=output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe archived failure rollouts without modifying student data"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_probe(args.config, args.registry, output=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
