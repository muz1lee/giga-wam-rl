import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence


EXPECTED_MODEL_CONFIG = {
    "_class_name": "CasualWorldActionTransformer_MoT",
    "in_channels": 48,
    "out_channels": 48,
    "in_action_channels": 16,
    "out_action_channels": 16,
    "num_layers": 30,
    "action_expert_dim": 1024,
    "num_embodiments": 2,
}

EXPECTED_CHECKPOINT_TOTAL_SIZE = 24_086_243_712
EXPECTED_CHECKPOINT_TENSOR_COUNT = 1_664
EXPECTED_TENSOR_SHAPES = {
    "state_encoder.in_proj.weight": (2, 16, 128),
    "state_encoder.in_proj.bias": (2, 128),
    "action_encoder.in_proj.weight": (2, 16, 128),
    "action_decoder.out_proj.weight": (2, 128, 16),
    "action_decoder.out_proj.bias": (2, 16),
    "patch_embedding.weight": (3072, 48, 1, 2, 2),
    "proj_out.weight": (192, 3072),
}


class ContractError(RuntimeError):
    """Raised when the pinned checkpoint violates the expected GWP0.5 contract."""


def validate_model_config(config: dict[str, Any]) -> None:
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            raise ContractError(
                f"model config {key} must be {expected!r}, got {actual!r}"
            )


def validate_loading_info(loading_info: dict[str, Any]) -> None:
    for key in (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    ):
        values = loading_info.get(key, [])
        if values:
            raise ContractError(f"strict checkpoint load reported {key}: {values}")


def expected_timestep_tokens() -> int:
    state_tokens = 1
    action_tokens = 48
    tokens_per_latent_frame = (24 // 2) * (20 // 2)
    return (
        state_tokens + tokens_per_latent_frame + action_tokens + tokens_per_latent_frame
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"required checkpoint file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    from safetensors import safe_open

    checkpoint = checkpoint.resolve(strict=True)
    config = _load_json(checkpoint / "config.json")
    validate_model_config(config)

    index = _load_json(checkpoint / "diffusion_pytorch_model.safetensors.index.json")
    total_size = index.get("metadata", {}).get("total_size")
    if total_size != EXPECTED_CHECKPOINT_TOTAL_SIZE:
        raise ContractError(
            "checkpoint tensor bytes must be "
            f"{EXPECTED_CHECKPOINT_TOTAL_SIZE}, got {total_size}"
        )

    weight_map = index.get("weight_map", {})
    if len(weight_map) != EXPECTED_CHECKPOINT_TENSOR_COUNT:
        raise ContractError(
            "checkpoint tensor count must be "
            f"{EXPECTED_CHECKPOINT_TENSOR_COUNT}, got {len(weight_map)}"
        )

    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != 3:
        raise ContractError(f"checkpoint must contain 3 shards, got {shard_names}")
    for shard_name in shard_names:
        shard_path = checkpoint / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise ContractError(f"checkpoint shard is missing or empty: {shard_path}")

    keys_by_shard: dict[str, list[str]] = {}
    for key in EXPECTED_TENSOR_SHAPES:
        shard_name = weight_map.get(key)
        if shard_name is None:
            raise ContractError(f"checkpoint tensor is missing from index: {key}")
        keys_by_shard.setdefault(shard_name, []).append(key)

    observed_shapes = {}
    for shard_name, keys in keys_by_shard.items():
        with safe_open(
            str(checkpoint / shard_name), framework="pt", device="cpu"
        ) as shard:
            for key in keys:
                shape = tuple(shard.get_slice(key).get_shape())
                observed_shapes[key] = shape
                expected_shape = EXPECTED_TENSOR_SHAPES[key]
                if shape != expected_shape:
                    raise ContractError(
                        f"checkpoint tensor {key} must have shape "
                        f"{expected_shape}, got {shape}"
                    )

    return {
        "checkpoint": str(checkpoint),
        "tensor_bytes": total_size,
        "tensor_count": len(weight_map),
        "shards": shard_names,
        "action_channels": config["in_action_channels"],
        "key_shapes": observed_shapes,
    }


def validate_upstream_checkout(project_root: Path, upstream_root: Path) -> str:
    upstream_config = project_root / "configs" / "upstreams.toml"
    with upstream_config.open("rb") as config_file:
        expected_revision = tomllib.load(config_file)["code"]["giga_world_policy_0_5"][
            "revision"
        ]

    revision = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise ContractError(
            f"upstream revision must be {expected_revision}, got {revision}"
        )

    status = subprocess.run(
        ["git", "-C", str(upstream_root), "status", "--porcelain=v1", "-uall"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ContractError("upstream checkout must be clean")
    return revision


def _add_upstream_import_paths(upstream_root: Path) -> None:
    paths = (
        upstream_root,
        upstream_root / "third_party" / "giga-models",
        upstream_root / "third_party" / "giga-train",
        upstream_root / "third_party" / "giga-datasets",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def strict_load_and_forward(
    checkpoint: Path,
    upstream_root: Path,
    *,
    device_name: str,
) -> dict[str, Any]:
    _add_upstream_import_paths(upstream_root)

    import torch

    from world_action_model.models import CasualWorldActionTransformer_MoT

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("the joint-forward smoke test requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    loaded = CasualWorldActionTransformer_MoT.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
    )
    model, loading_info = loaded
    validate_loading_info(loading_info)
    model.eval()
    model.to(device)

    embodiment_id = torch.zeros(1, dtype=torch.long, device=device)
    negative_32d_error = None
    try:
        with torch.inference_mode():
            model.encode_action_tokens(
                torch.zeros(1, 1, 32, dtype=torch.bfloat16, device=device),
                torch.zeros(1, 48, 32, dtype=torch.bfloat16, device=device),
                embodiment_id=embodiment_id,
            )
    except RuntimeError as error:
        negative_32d_error = type(error).__name__
    if negative_32d_error is None:
        raise ContractError(
            "the pinned 16D checkpoint unexpectedly accepted 32D actions"
        )

    state = torch.zeros(1, 1, 16, dtype=torch.bfloat16, device=device)
    action = torch.zeros(1, 48, 16, dtype=torch.bfloat16, device=device)
    ref_latents = torch.zeros(1, 48, 1, 24, 20, dtype=torch.bfloat16, device=device)
    noisy_latents = torch.zeros_like(ref_latents)
    prompt_embeds = torch.zeros(1, 64, 4096, dtype=torch.bfloat16, device=device)

    timestep = torch.zeros(
        1, expected_timestep_tokens(), dtype=torch.long, device=device
    )
    ref_end = 1 + (24 // 2) * (20 // 2)
    action_end = ref_end + 48
    timestep[:, ref_end:action_end] = 500
    timestep[:, action_end:] = 500

    with torch.inference_mode():
        output = model(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            state=state,
            action=action,
            embodiment_id=embodiment_id,
            return_dict=True,
        )

    visual_pred = output["sample"]
    action_pred = output["action_pred"]
    expected_visual_shape = (1, 48, 1, 24, 20)
    expected_action_shape = (1, 48, 16)
    if tuple(visual_pred.shape) != expected_visual_shape:
        raise ContractError(
            f"visual output must have shape {expected_visual_shape}, "
            f"got {tuple(visual_pred.shape)}"
        )
    if tuple(action_pred.shape) != expected_action_shape:
        raise ContractError(
            f"action output must have shape {expected_action_shape}, "
            f"got {tuple(action_pred.shape)}"
        )
    if not torch.isfinite(visual_pred).all().item():
        raise ContractError("visual output contains NaN or Inf")
    if not torch.isfinite(action_pred).all().item():
        raise ContractError("action output contains NaN or Inf")

    return {
        "device": str(device),
        "dtype": "bfloat16",
        "negative_32d_error": negative_32d_error,
        "visual_shape": list(visual_pred.shape),
        "action_shape": list(action_pred.shape),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_reserved(device) / (1024**3), 3
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Validate and smoke-test the pinned GigaWorld-Policy-0.5 transformer"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("GWP05_TRANSFORMER_PRETRAINED"),
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=project_root / "external" / "giga-world-policy",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--manifest-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.checkpoint is None:
        parser.error("--checkpoint or GWP05_TRANSFORMER_PRETRAINED is required")

    revision = validate_upstream_checkout(project_root, arguments.upstream_root)
    manifest = validate_checkpoint_manifest(arguments.checkpoint)
    print(
        json.dumps(
            {"status": "manifest_ok", "code_revision": revision, **manifest},
            sort_keys=True,
        )
    )
    if arguments.manifest_only:
        return 0

    forward = strict_load_and_forward(
        arguments.checkpoint,
        arguments.upstream_root,
        device_name=arguments.device,
    )
    print(json.dumps({"status": "forward_ok", **forward}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
