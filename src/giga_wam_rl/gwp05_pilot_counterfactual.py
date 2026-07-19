import argparse
import gc
import hashlib
import html
import json
import re
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from giga_wam_rl.gwp05_future_sampler import (
    FUTURE_LATENT_SHAPE,
    VISUAL_FLOW_SHIFT,
    future_only_denoise,
)
from giga_wam_rl.gwp05_pilot_inputs import (
    build_action_state_conditions,
    compose_t_layout,
    make_additive_action_counterfactual,
    rgb_frames_to_vae_video,
)
from giga_wam_rl.gwp05_smoke import (
    ContractError,
    _add_upstream_import_paths,
    validate_checkpoint_manifest,
    validate_loading_info,
    validate_upstream_checkout,
)
from giga_wam_rl.gwp05_vae_smoke import validate_vae_manifest
from giga_wam_rl.raw_hdf5_probe import _decode_jpeg, _find_registered_asset
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


DELTA_MASK = np.asarray(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ],
    dtype=bool,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as input_file:
        return tomllib.load(input_file)


def _resolve_model_paths(
    upstream_config: Path,
    *,
    checkpoint: Path | None,
    base_model: Path | None,
) -> tuple[Path, Path]:
    config = _load_toml(upstream_config)
    if checkpoint is None:
        checkpoint = Path(config["models"]["giga_world_policy_0_5_transformer"]["target"])
    if base_model is None:
        base_model = Path(config["models"]["wan_2_2_ti2v_5b_diffusers"]["target"])
    return checkpoint, base_model


def _load_raw_window(
    data_config: dict[str, Any],
    *,
    registry: dict[str, Any],
    episode_id: int,
    start: int,
) -> dict[str, Any]:
    import h5py

    dataset_config = data_config["dataset"]
    alignment = data_config["alignment"]
    camera_map = data_config["cameras"]["map"]
    asset = _find_registered_asset(registry, dataset_config["asset_name"])
    source_root = Path(dataset_config["source_root"]).resolve(strict=True)
    if source_root != Path(asset["path"]).resolve(strict=True):
        raise ContractError("raw source does not match the registered read-only asset")

    horizon = int(alignment["action_horizon"])
    frame_offsets = [int(offset) for offset in alignment["frame_offsets"]]
    if frame_offsets != [0, 12, 24, 36, 48] or horizon != 48:
        raise ContractError("pilot runner requires the validated H48 visual contract")
    episode_path = source_root / f"episode{episode_id}.hdf5"
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)

    with h5py.File(episode_path, "r") as episode_file:
        vector = np.asarray(
            episode_file[alignment["state_key"]], dtype=np.float32
        )
        if start < 0 or start + horizon >= vector.shape[0]:
            raise ValueError(
                f"window [{start}, {start + horizon}] is outside episode length "
                f"{vector.shape[0]}"
            )
        state = vector[start].copy()
        actions = vector[start + 1 : start + horizon + 1].copy()
        composite_frames = []
        for offset in frame_offsets:
            camera_images = {
                model_key: _decode_jpeg(episode_file[source_key][start + offset])
                for model_key, source_key in camera_map.items()
            }
            composite_frames.append(
                compose_t_layout(
                    camera_images["cam_high"],
                    camera_images["cam_left_wrist"],
                    camera_images["cam_right_wrist"],
                )
            )
    if state.shape != (14,) or actions.shape != (48, 14):
        raise ContractError(
            f"raw vectors violate the 14D/H48 contract: {state.shape}, {actions.shape}"
        )
    return {
        "episode_path": episode_path,
        "episode_length": int(vector.shape[0]),
        "state": state,
        "actions": actions,
        "composite_frames": np.stack(composite_frames),
        "frame_offsets": frame_offsets,
    }


def _load_norm_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("model_dim") != 16 or metadata.get("action_horizon") != 48:
        raise ContractError("norm stats must declare model_dim=16 and action_horizon=48")
    stats = payload.get("norm_stats", {})
    if not {"observation.state", "action"}.issubset(stats):
        raise ContractError("norm stats are missing state or action features")
    return payload


def _prompt_clean(text: str) -> str:
    try:
        import ftfy
    except ImportError as error:
        raise ContractError("ftfy==6.3.1 is required for Wan prompt cleaning") from error
    cleaned = ftfy.fix_text(text)
    cleaned = html.unescape(html.unescape(cleaned))
    return re.sub(r"\s+", " ", cleaned).strip()


def _encode_prompt(
    base_model: Path,
    prompt: str,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(
        base_model / "tokenizer", local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model / "text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval().requires_grad_(False).to(device)
    tokens = tokenizer(
        [_prompt_clean(prompt)],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.inference_mode():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    prompt_embeds = hidden[:, :64].to(dtype=dtype)
    prompt_embeds = prompt_embeds * attention_mask[:, :64, None].to(dtype)
    if tuple(prompt_embeds.shape) != (1, 64, 4096):
        raise ContractError(
            f"prompt embedding must have shape [1,64,4096], got {prompt_embeds.shape}"
        )
    del text_encoder, tokenizer, tokens, input_ids, attention_mask, hidden
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_embeds


def _load_vae(base_model: Path, *, device: Any, dtype: Any) -> Any:
    from diffusers.models import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    )
    return vae.eval().requires_grad_(False).to(device)


def _normalized_reference_latent(
    vae: Any,
    reference_rgb: np.ndarray,
    *,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, Any]:
    import torch

    reference_video = rgb_frames_to_vae_video([reference_rgb]).to(
        device=device, dtype=dtype
    )
    with torch.inference_mode():
        raw_latent = vae.encode(reference_video).latent_dist.mode()
    mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=torch.float32
    ).view(1, 48, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std, device=device, dtype=torch.float32
    ).view(1, 48, 1, 1, 1)
    normalized = ((raw_latent.float() - mean) / std).to(dtype=dtype)
    expected_shape = (1, 48, 1, 24, 20)
    if tuple(normalized.shape) != expected_shape:
        raise ContractError(
            f"reference latent must have shape {expected_shape}, got {normalized.shape}"
        )
    return normalized, mean, std


def _load_transformer(
    checkpoint: Path,
    upstream_root: Path,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    _add_upstream_import_paths(upstream_root)
    from world_action_model.models import CasualWorldActionTransformer_MoT

    model, loading_info = CasualWorldActionTransformer_MoT.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
    )
    validate_loading_info(loading_info)
    return model.eval().requires_grad_(False).to(device)


def _decoded_video_to_uint8(video: Any) -> np.ndarray:
    video = video.float().clamp(-1.0, 1.0)
    uint8 = ((video + 1.0) * 127.5).round().byte()
    return uint8.permute(0, 2, 3, 4, 1).cpu().numpy()


def _save_frame_rows(
    output_dir: Path,
    *,
    ground_truth: np.ndarray,
    generated_demo: np.ndarray,
    generated_counterfactual: np.ndarray,
    frame_offsets: list[int],
) -> None:
    rows = {
        "ground_truth_demo": ground_truth,
        "generated_demo_action": generated_demo,
        "generated_counterfactual": generated_counterfactual,
    }
    for row_name, frames in rows.items():
        row_dir = output_dir / row_name
        row_dir.mkdir(parents=True)
        for offset, frame in zip(frame_offsets, frames, strict=True):
            Image.fromarray(frame).save(row_dir / f"offset_{offset:02d}.png")

    label_width = 196
    frame_height, frame_width = ground_truth.shape[1:3]
    sheet = Image.new(
        "RGB",
        (label_width + frame_width * len(frame_offsets), frame_height * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (row_name, frames) in enumerate(rows.items()):
        y = row_index * frame_height
        draw.text((8, y + 8), row_name, fill="black")
        for column_index, (offset, frame) in enumerate(
            zip(frame_offsets, frames, strict=True)
        ):
            x = label_width + column_index * frame_width
            sheet.paste(Image.fromarray(frame), (x, y))
            draw.text((x + 5, y + 5), f"t+{offset}", fill="white", stroke_width=2, stroke_fill="black")
    sheet.save(output_dir / "comparison_contact_sheet.png")


def run_counterfactual_smoke(
    *,
    project_root: Path,
    registry_path: Path,
    data_config_path: Path,
    norm_stats_path: Path,
    checkpoint: Path,
    base_model: Path,
    upstream_root: Path,
    output_dir: Path,
    device_name: str,
    episode_id: int,
    start: int,
    action_dimension: int,
    additive_offset: float,
    num_inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    registry = load_registry(registry_path)
    validate_registry(registry)
    output_dir = validate_output_root(
        output_dir,
        artifact_root=Path(registry["workspace"]["artifact_root"]),
        protected_roots=tuple(
            Path(path) for path in registry["workspace"]["protected_roots"]
        ),
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual run: {output_dir}")
    code_revision = validate_upstream_checkout(project_root, upstream_root)
    checkpoint_manifest = validate_checkpoint_manifest(checkpoint)
    vae_manifest = validate_vae_manifest(base_model)
    data_config = _load_toml(data_config_path)
    norm_payload = _load_norm_stats(norm_stats_path)
    window = _load_raw_window(
        data_config,
        registry=registry,
        episode_id=episode_id,
        start=start,
    )

    instruction = data_config["dataset"]["canonical_instruction"]
    state = window["state"]
    demo_actions = window["actions"]
    counterfactual_actions = make_additive_action_counterfactual(
        demo_actions,
        action_dimension=action_dimension,
        additive_offset=additive_offset,
    )
    norm_stats = norm_payload["norm_stats"]
    normalized_state, normalized_demo_action = build_action_state_conditions(
        state,
        demo_actions,
        state_stats=norm_stats["observation.state"],
        action_stats=norm_stats["action"],
        delta_mask=DELTA_MASK,
        model_dimensions=16,
    )
    repeated_state, normalized_counterfactual_action = (
        build_action_state_conditions(
            state,
            counterfactual_actions,
            state_stats=norm_stats["observation.state"],
            action_stats=norm_stats["action"],
            delta_mask=DELTA_MASK,
            model_dimensions=16,
        )
    )
    if not np.array_equal(normalized_state, repeated_state):
        raise ContractError("paired counterfactual state conditions differ")

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("counterfactual smoke requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    dtype = torch.bfloat16
    started = time.perf_counter()
    prompt_embeds = _encode_prompt(
        base_model, instruction, device=device, dtype=dtype
    )
    vae = _load_vae(base_model, device=device, dtype=dtype)
    ref_latent, latent_mean, latent_std = _normalized_reference_latent(
        vae,
        window["composite_frames"][0],
        device=device,
        dtype=dtype,
    )
    transformer = _load_transformer(
        checkpoint, upstream_root, device=device, dtype=dtype
    )

    batch_size = 2
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(
        1,
        *FUTURE_LATENT_SHAPE,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    ref_batch = ref_latent.repeat(batch_size, 1, 1, 1, 1)
    state_batch = torch.from_numpy(
        np.repeat(normalized_state[None], batch_size, axis=0)
    ).to(device=device, dtype=dtype)
    action_batch = torch.from_numpy(
        np.stack((normalized_demo_action, normalized_counterfactual_action))
    ).to(device=device, dtype=dtype)
    prompt_batch = prompt_embeds.repeat(batch_size, 1, 1)
    embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)
    future_latents, timesteps = future_only_denoise(
        transformer,
        ref_latents=ref_batch,
        initial_future=initial_noise.repeat(batch_size, 1, 1, 1, 1),
        state=state_batch,
        action=action_batch,
        prompt_embeds=prompt_batch,
        embodiment_id=embodiment_id,
        num_inference_steps=num_inference_steps,
    )
    latent_difference = (
        future_latents[0].float() - future_latents[1].float()
    ).abs()
    joint_latents = torch.cat((ref_batch, future_latents), dim=2)
    raw_joint_latents = (
        joint_latents.float() * latent_std + latent_mean
    ).to(dtype=dtype)
    decoded_samples = []
    with torch.inference_mode():
        for sample_index in range(batch_size):
            decoded_samples.append(
                vae.decode(
                    raw_joint_latents[sample_index : sample_index + 1],
                    return_dict=False,
                )[0]
            )
    decoded = torch.cat(decoded_samples, dim=0)
    generated = _decoded_video_to_uint8(decoded)
    elapsed = time.perf_counter() - started

    ground_truth = window["composite_frames"]
    generated_demo = generated[0]
    generated_counterfactual = generated[1]
    future_pixel_difference = np.abs(
        generated_demo[1:].astype(np.float32)
        - generated_counterfactual[1:].astype(np.float32)
    )
    demo_ground_truth_error = np.abs(
        generated_demo[1:].astype(np.float32)
        - ground_truth[1:].astype(np.float32)
    )
    reference_pair_error = np.abs(
        generated_demo[0].astype(np.float32)
        - generated_counterfactual[0].astype(np.float32)
    )

    output_dir.mkdir(parents=True)
    _save_frame_rows(
        output_dir,
        ground_truth=ground_truth,
        generated_demo=generated_demo,
        generated_counterfactual=generated_counterfactual,
        frame_offsets=window["frame_offsets"],
    )
    np.savez_compressed(
        output_dir / "conditions.npz",
        raw_state=state,
        raw_demo_action=demo_actions,
        raw_counterfactual_action=counterfactual_actions,
        normalized_state=normalized_state,
        normalized_demo_action=normalized_demo_action,
        normalized_counterfactual_action=normalized_counterfactual_action,
        ground_truth_composite_frames=ground_truth,
    )
    normalized_action_difference = np.abs(
        normalized_counterfactual_action - normalized_demo_action
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "paired_counterfactual_smoke_ok",
        "scope": (
            "structural action-conditioning smoke; not evidence of calibrated "
            "failure prediction"
        ),
        "source": {
            "episode_path": str(window["episode_path"]),
            "episode_id": episode_id,
            "episode_length": window["episode_length"],
            "start": start,
            "frame_offsets": window["frame_offsets"],
            "state_sha256": _sha256_array(state),
            "demo_action_sha256": _sha256_array(demo_actions),
            "ground_truth_frames_sha256": _sha256_array(ground_truth),
            "instruction": instruction,
        },
        "counterfactual": {
            "action_dimension": action_dimension,
            "additive_raw_offset": additive_offset,
            "raw_joint_units": "source joint-target units; arm dimensions are radians",
            "normalized_mean_abs_difference": float(
                normalized_action_difference.mean()
            ),
            "normalized_max_abs_difference": float(
                normalized_action_difference.max()
            ),
        },
        "model": {
            "code_revision": code_revision,
            "checkpoint": str(checkpoint),
            "checkpoint_tensor_count": checkpoint_manifest["tensor_count"],
            "base_model": str(base_model),
            "vae_weights_bytes": vae_manifest["weights_bytes"],
            "dtype": "bfloat16",
            "device": str(device),
            "prompt_shape": list(prompt_embeds.shape),
            "visual_flow_shift": VISUAL_FLOW_SHIFT,
            "num_inference_steps": num_inference_steps,
            "scheduler_timesteps": timesteps,
            "seed": seed,
            "vae_decode_batch_size": 1,
        },
        "normalization": {
            "path": str(norm_stats_path),
            "sha256": _sha256_file(norm_stats_path),
            "model_dimensions": 16,
            "physical_dimensions": 14,
            "clamp": False,
        },
        "metrics": {
            "future_latent_mean_abs_difference": float(
                latent_difference.mean().item()
            ),
            "future_latent_max_abs_difference": float(
                latent_difference.max().item()
            ),
            "future_pixel_mean_abs_difference_0_255": float(
                future_pixel_difference.mean()
            ),
            "future_pixel_per_offset_mean_abs_difference_0_255": [
                float(value)
                for value in future_pixel_difference.mean(axis=(1, 2, 3))
            ],
            "demo_future_vs_ground_truth_mean_abs_error_0_255": float(
                demo_ground_truth_error.mean()
            ),
            "paired_decoded_reference_max_abs_difference_0_255": float(
                reference_pair_error.max()
            ),
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 3
            ),
            "total_wall_seconds": round(elapsed, 3),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "contact_sheet": "comparison_contact_sheet.png",
            "conditions": "conditions.npz",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run a real Place Bread paired counterfactual future smoke"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "configs" / "assets.server.toml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=project_root / "configs" / "datasets" / "place_bread_raw_hdf5.toml",
    )
    parser.add_argument(
        "--upstream-config",
        type=Path,
        default=project_root / "configs" / "upstreams.toml",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=project_root / "external" / "giga-world-policy",
    )
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--action-dimension", type=int, default=0)
    parser.add_argument("--additive-offset", type=float, default=0.5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args(argv)
    checkpoint, base_model = _resolve_model_paths(
        arguments.upstream_config,
        checkpoint=arguments.checkpoint,
        base_model=arguments.base_model,
    )
    manifest = run_counterfactual_smoke(
        project_root=project_root,
        registry_path=arguments.registry,
        data_config_path=arguments.data_config,
        norm_stats_path=arguments.norm_stats,
        checkpoint=checkpoint,
        base_model=base_model,
        upstream_root=arguments.upstream_root,
        output_dir=arguments.output_dir,
        device_name=arguments.device,
        episode_id=arguments.episode_id,
        start=arguments.start,
        action_dimension=arguments.action_dimension,
        additive_offset=arguments.additive_offset,
        num_inference_steps=arguments.num_inference_steps,
        seed=arguments.seed,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
