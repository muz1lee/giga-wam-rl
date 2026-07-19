from typing import Any


def future_flow_batch(
    visual_latents: Any,
    *,
    sigma: Any,
    noise: Any,
) -> tuple[Any, Any, Any]:
    if visual_latents.ndim != 5:
        raise ValueError("visual latents must have shape [B, C, T, H, W]")
    if visual_latents.shape[2] < 2:
        raise ValueError("visual latents must contain reference and future frames")
    future_latents = visual_latents[:, :, 1:]
    if noise.shape != future_latents.shape:
        raise ValueError("noise shape must equal future latent shape")
    noisy_future = noise * sigma + future_latents * (1 - sigma)
    future_target = noise - future_latents
    return future_latents, noisy_future, future_target


def build_clean_action_timestep(
    visual_timestep: Any,
    *,
    num_state_tokens: int,
    num_ref_tokens: int,
    num_action_tokens: int,
    num_future_tokens: int,
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    token_counts = (
        num_state_tokens,
        num_ref_tokens,
        num_action_tokens,
        num_future_tokens,
    )
    if any(count <= 0 for count in token_counts):
        raise ValueError("state, reference, action, and future need positive tokens")
    batch_size = int(visual_timestep.shape[0])
    timestep = torch.zeros(
        batch_size,
        sum(token_counts),
        dtype=dtype,
        device=device,
    )
    future_start = num_state_tokens + num_ref_tokens + num_action_tokens
    timestep[:, future_start:] = visual_timestep[:, None].to(dtype=dtype)
    return timestep
