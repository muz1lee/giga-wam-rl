import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


EXPECTED_VAE_CONFIG = {
    "_class_name": "AutoencoderKLWan",
    "z_dim": 48,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "patch_size": 2,
}


class VAEContractError(RuntimeError):
    """Raised when the pinned Wan VAE violates the GWP0.5 latent contract."""


def validate_vae_config(config: dict[str, Any]) -> None:
    for key, expected in EXPECTED_VAE_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            raise VAEContractError(
                f"VAE config {key} must be {expected!r}, got {actual!r}"
            )

    for key in ("latents_mean", "latents_std"):
        values = config.get(key)
        if not isinstance(values, list) or len(values) != 48:
            raise VAEContractError(f"VAE config {key} must contain 48 values")
    if any(float(value) == 0.0 for value in config["latents_std"]):
        raise VAEContractError("VAE config latents_std must be nonzero")


def expected_latent_shape(
    num_frames: int,
    *,
    batch_size: int = 1,
    height: int = 384,
    width: int = 320,
) -> tuple[int, ...]:
    if num_frames < 1 or (num_frames - 1) % 4 != 0:
        raise VAEContractError("num_frames must be positive and congruent to 1 mod 4")
    return (
        batch_size,
        48,
        (num_frames - 1) // 4 + 1,
        height // 16,
        width // 16,
    )


def expected_postprocessed_shape(num_frames: int) -> tuple[int, ...]:
    return (1, num_frames, 3, 384, 320)


def validate_vae_manifest(base_model: Path) -> dict[str, Any]:
    base_model = base_model.resolve(strict=True)
    config_path = base_model / "vae" / "config.json"
    weights_path = base_model / "vae" / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file():
        raise VAEContractError(f"VAE config is missing: {config_path}")
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise VAEContractError(f"VAE weights are missing or empty: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_vae_config(config)
    return {
        "base_model": str(base_model),
        "weights_bytes": weights_path.stat().st_size,
        "z_dim": config["z_dim"],
        "spatial_factor": config["scale_factor_spatial"],
        "temporal_factor": config["scale_factor_temporal"],
    }


def encode_decode_smoke(base_model: Path, *, device_name: str) -> dict[str, Any]:
    import torch
    from diffusers.models import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise VAEContractError("the VAE smoke test requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    vae.eval()
    vae.to(device)

    video = torch.zeros(1, 3, 5, 384, 320, dtype=torch.bfloat16, device=device)
    if video.min().item() < -1.0 or video.max().item() > 1.0:
        raise VAEContractError("VAE input must be normalized to [-1, 1]")
    with torch.inference_mode():
        raw_latents = vae.encode(video).latent_dist.mode()
        single_frame_latents = vae.encode(video[:, :, :1]).latent_dist.mode()

    expected_shape = expected_latent_shape(5)
    if tuple(raw_latents.shape) != expected_shape:
        raise VAEContractError(
            f"five-frame latent must have shape {expected_shape}, "
            f"got {tuple(raw_latents.shape)}"
        )
    expected_single_frame_shape = expected_latent_shape(1)
    if tuple(single_frame_latents.shape) != expected_single_frame_shape:
        raise VAEContractError(
            f"single-frame latent must have shape {expected_single_frame_shape}, "
            f"got {tuple(single_frame_latents.shape)}"
        )

    mean = torch.tensor(
        vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, 48, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(
        1, 48, 1, 1, 1
    )
    normalized = (raw_latents.float() - mean) / std
    recovered = normalized * std + mean
    if not torch.isfinite(normalized).all().item():
        raise VAEContractError("normalized VAE latents contain NaN or Inf")
    if not torch.allclose(recovered, raw_latents.float(), atol=1e-5, rtol=1e-5):
        raise VAEContractError("VAE normalization is not numerically invertible")

    with torch.inference_mode():
        reconstruction = vae.decode(recovered.to(dtype=vae.dtype), return_dict=False)[0]
    expected_reconstruction_shape = (1, 3, 5, 384, 320)
    if tuple(reconstruction.shape) != expected_reconstruction_shape:
        raise VAEContractError(
            f"VAE reconstruction must have shape {expected_reconstruction_shape}, "
            f"got {tuple(reconstruction.shape)}"
        )
    if not torch.isfinite(reconstruction).all().item():
        raise VAEContractError("VAE reconstruction contains NaN or Inf")
    if reconstruction.min().item() < -1.001 or reconstruction.max().item() > 1.001:
        raise VAEContractError("VAE reconstruction must stay within [-1, 1]")

    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    postprocessed = video_processor.postprocess_video(reconstruction, output_type="pt")
    expected_post_shape = expected_postprocessed_shape(5)
    if tuple(postprocessed.shape) != expected_post_shape:
        raise VAEContractError(
            f"postprocessed video must have shape {expected_post_shape}, "
            f"got {tuple(postprocessed.shape)}"
        )
    if postprocessed.min().item() < 0.0 or postprocessed.max().item() > 1.0:
        raise VAEContractError("postprocessed video must stay within [0, 1]")

    return {
        "device": str(device),
        "dtype": "bfloat16",
        "input_shape": list(video.shape),
        "latent_shape": list(raw_latents.shape),
        "single_frame_latent_shape": list(single_frame_latents.shape),
        "reference_shape": list(raw_latents[:, :, :1].shape),
        "future_shape": list(raw_latents[:, :, 1:].shape),
        "reconstruction_shape": list(reconstruction.shape),
        "postprocessed_shape": list(postprocessed.shape),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_reserved(device) / (1024**3), 3
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pinned Wan2.2 VAE contract used by GWP0.5"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=os.environ.get("GWP_WAN_PRETRAINED"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--manifest-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.base_model is None:
        parser.error("--base-model or GWP_WAN_PRETRAINED is required")

    manifest = validate_vae_manifest(arguments.base_model)
    print(json.dumps({"status": "vae_manifest_ok", **manifest}, sort_keys=True))
    if arguments.manifest_only:
        return 0

    smoke = encode_decode_smoke(arguments.base_model, device_name=arguments.device)
    print(json.dumps({"status": "vae_smoke_ok", **smoke}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
