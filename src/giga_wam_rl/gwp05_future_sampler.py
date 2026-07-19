import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from giga_wam_rl.gwp05_smoke import (
    ContractError,
    validate_checkpoint_manifest,
    validate_loading_info,
    validate_upstream_checkout,
)


VISUAL_FLOW_SHIFT = 2.0
FUTURE_LATENT_SHAPE = (48, 1, 24, 20)


def build_future_only_timestep(
    *,
    batch_size: int,
    visual_timestep: int,
    num_state_tokens: int,
    num_action_tokens: int,
    num_ref_tokens: int,
    num_future_tokens: int,
) -> list[list[int]]:
    counts = (
        batch_size,
        num_state_tokens,
        num_action_tokens,
        num_ref_tokens,
        num_future_tokens,
    )
    if any(count <= 0 for count in counts):
        raise ValueError("batch and token counts must be positive")
    if not 0 <= visual_timestep <= 1000:
        raise ValueError("visual_timestep must be between 0 and 1000")

    clean_prefix_tokens = (
        num_state_tokens + num_ref_tokens + num_action_tokens
    )
    row = [0] * clean_prefix_tokens + [visual_timestep] * num_future_tokens
    return [row.copy() for _ in range(batch_size)]


def create_visual_scheduler() -> Any:
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=VISUAL_FLOW_SHIFT,
    )


def future_only_denoise(
    model: Any,
    *,
    ref_latents: Any,
    initial_future: Any,
    state: Any,
    action: Any,
    prompt_embeds: Any,
    embodiment_id: Any,
    num_inference_steps: int,
) -> tuple[Any, list[float]]:
    import torch

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    batch_size = ref_latents.shape[0]
    expected_ref = (batch_size, 48, 1, 24, 20)
    expected_future = (batch_size, *FUTURE_LATENT_SHAPE)
    if tuple(ref_latents.shape) != expected_ref:
        raise ContractError(
            f"reference latents must have shape {expected_ref}, "
            f"got {tuple(ref_latents.shape)}"
        )
    if tuple(initial_future.shape) != expected_future:
        raise ContractError(
            f"initial future must have shape {expected_future}, "
            f"got {tuple(initial_future.shape)}"
        )
    if tuple(state.shape) != (batch_size, 1, 16):
        raise ContractError("state must have shape [B, 1, 16]")
    if tuple(action.shape) != (batch_size, 48, 16):
        raise ContractError("action must have shape [B, 48, 16]")
    if prompt_embeds.shape[0] != batch_size or prompt_embeds.shape[-1] != 4096:
        raise ContractError("prompt_embeds must have shape [B, L, 4096]")

    scheduler = create_visual_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=ref_latents.device)
    future = initial_future.clone()
    num_ref_tokens = (ref_latents.shape[-2] // 2) * (ref_latents.shape[-1] // 2)
    num_future_tokens = (
        initial_future.shape[2]
        * (initial_future.shape[-2] // 2)
        * (initial_future.shape[-1] // 2)
    )
    used_timesteps = []

    with torch.inference_mode():
        for scheduler_timestep in scheduler.timesteps:
            visual_timestep = int(round(float(scheduler_timestep.item())))
            used_timesteps.append(float(scheduler_timestep.item()))
            timestep_rows = build_future_only_timestep(
                batch_size=batch_size,
                visual_timestep=visual_timestep,
                num_state_tokens=state.shape[1],
                num_action_tokens=action.shape[1],
                num_ref_tokens=num_ref_tokens,
                num_future_tokens=num_future_tokens,
            )
            timestep = torch.tensor(
                timestep_rows,
                device=ref_latents.device,
                dtype=ref_latents.dtype,
            )
            output = model(
                ref_latents=ref_latents,
                noisy_latents=future,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                state=state,
                action=action,
                embodiment_id=embodiment_id,
                return_dict=True,
            )
            visual_velocity = output["sample"][:, :, 1:]
            if tuple(visual_velocity.shape) != expected_future:
                raise ContractError(
                    f"future velocity must have shape {expected_future}, "
                    f"got {tuple(visual_velocity.shape)}"
                )
            future = scheduler.step(
                visual_velocity,
                scheduler_timestep,
                future,
                return_dict=False,
            )[0]

    if not torch.isfinite(future).all().item():
        raise ContractError("future sampler produced NaN or Inf")
    return future, used_timesteps


def synthetic_action_conditioning_smoke(
    checkpoint: Path,
    upstream_root: Path,
    *,
    device_name: str,
    num_inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    from giga_wam_rl.gwp05_smoke import _add_upstream_import_paths

    _add_upstream_import_paths(upstream_root)

    import torch

    from world_action_model.models import CasualWorldActionTransformer_MoT

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("the future-only smoke test requires a CUDA device")
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

    batch_size = 2
    generator = torch.Generator(device=device).manual_seed(seed)
    base_noise = torch.randn(
        1,
        *FUTURE_LATENT_SHAPE,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    initial_future = base_noise.repeat(batch_size, 1, 1, 1, 1)
    ref_latents = torch.zeros(
        batch_size, 48, 1, 24, 20, device=device, dtype=torch.bfloat16
    )
    state = torch.zeros(
        batch_size, 1, 16, device=device, dtype=torch.bfloat16
    )
    action = torch.zeros(
        batch_size, 48, 16, device=device, dtype=torch.bfloat16
    )
    action[1, :, 0] = 0.5
    prompt_embeds = torch.zeros(
        batch_size, 64, 4096, device=device, dtype=torch.bfloat16
    )
    embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)

    future, timesteps = future_only_denoise(
        model,
        ref_latents=ref_latents,
        initial_future=initial_future,
        state=state,
        action=action,
        prompt_embeds=prompt_embeds,
        embodiment_id=embodiment_id,
        num_inference_steps=num_inference_steps,
    )
    conditioned_difference = (future[0].float() - future[1].float()).abs().mean()
    if conditioned_difference.item() <= 0:
        raise ContractError("different clean actions produced identical futures")

    return {
        "device": str(device),
        "dtype": "bfloat16",
        "batch_size": batch_size,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "scheduler": "FlowMatchEulerDiscreteScheduler",
        "visual_flow_shift": VISUAL_FLOW_SHIFT,
        "scheduler_timesteps": timesteps,
        "future_shape": list(future.shape),
        "action_conditioned_mean_abs_difference": conditioned_difference.item(),
        "initial_future_mean_abs": base_noise.float().abs().mean().item(),
        "final_future_mean_abs": future.float().abs().mean().item(),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_reserved(device) / (1024**3), 3
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Smoke-test GWP-0.5 action-conditioned future-only denoising"
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args(argv)
    if arguments.checkpoint is None:
        parser.error("--checkpoint or GWP05_TRANSFORMER_PRETRAINED is required")

    revision = validate_upstream_checkout(project_root, arguments.upstream_root)
    checkpoint_manifest = validate_checkpoint_manifest(arguments.checkpoint)
    result = synthetic_action_conditioning_smoke(
        arguments.checkpoint,
        arguments.upstream_root,
        device_name=arguments.device,
        num_inference_steps=arguments.num_inference_steps,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "status": "future_only_smoke_ok",
                "code_revision": revision,
                "checkpoint_tensor_count": checkpoint_manifest["tensor_count"],
                **result,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
