from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


COMPOSITE_WIDTH = 320
COMPOSITE_HEIGHT = 384


def cover_resize_crop_geometry(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    dimensions = (source_width, source_height, target_width, target_height)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("source and target dimensions must be positive")
    if target_height / source_height < target_width / source_width:
        resized_height = round(target_width / source_width * source_height)
        resized_width = target_width
    else:
        resized_height = target_height
        resized_width = round(target_height / source_height * source_width)
    crop_left = (resized_width - target_width) // 2
    crop_top = (resized_height - target_height) // 2
    return resized_width, resized_height, crop_left, crop_top


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape [H, W, 3], got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"RGB image must be uint8, got {image.dtype}")
    return image


def resize_center_crop_rgb(
    image: np.ndarray, *, target_width: int, target_height: int
) -> np.ndarray:
    image = _validate_rgb(image)
    source_height, source_width = image.shape[:2]
    resized_width, resized_height, crop_left, crop_top = (
        cover_resize_crop_geometry(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )
    )
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    cropped = resized.crop(
        (
            crop_left,
            crop_top,
            crop_left + target_width,
            crop_top + target_height,
        )
    )
    return np.asarray(cropped, dtype=np.uint8)


def compose_t_layout(
    front: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    top_height = COMPOSITE_HEIGHT // 2
    bottom_height = COMPOSITE_HEIGHT - top_height
    left_width = COMPOSITE_WIDTH // 2
    right_width = COMPOSITE_WIDTH - left_width
    front_crop = resize_center_crop_rgb(
        front,
        target_width=COMPOSITE_WIDTH,
        target_height=top_height,
    )
    left_crop = resize_center_crop_rgb(
        left_wrist,
        target_width=left_width,
        target_height=bottom_height,
    )
    right_crop = resize_center_crop_rgb(
        right_wrist,
        target_width=right_width,
        target_height=bottom_height,
    )
    bottom = np.concatenate((left_crop, right_crop), axis=1)
    return np.concatenate((front_crop, bottom), axis=0)


def _quantile_normalize(
    values: np.ndarray,
    stats: Mapping[str, Any],
    *,
    effective_dimensions: int,
    output_dimensions: int,
    clamp: bool,
) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    high = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    if min(low.size, high.size) < effective_dimensions:
        raise ValueError("q01/q99 stats do not cover the physical dimensions")
    scale = high[:effective_dimensions] - low[:effective_dimensions]
    if np.any(scale <= 0):
        raise ValueError("q99 must be greater than q01 for every physical dimension")
    output = np.zeros((*values.shape[:-1], output_dimensions), dtype=np.float32)
    output[..., :effective_dimensions] = (
        (values[..., :effective_dimensions] - low[:effective_dimensions])
        / scale
        * 2.0
        - 1.0
    )
    if clamp:
        np.clip(output, -1.0, 1.0, out=output)
    return output


def build_action_state_conditions(
    state: np.ndarray,
    future_actions: np.ndarray,
    *,
    state_stats: Mapping[str, Any],
    action_stats: Mapping[str, Any],
    delta_mask: np.ndarray,
    model_dimensions: int = 16,
    clamp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    future_actions = np.asarray(future_actions, dtype=np.float32)
    delta_mask = np.asarray(delta_mask, dtype=bool).reshape(-1)
    effective_dimensions = delta_mask.size
    if state.shape != (effective_dimensions,):
        raise ValueError(
            f"state must have shape {(effective_dimensions,)}, got {state.shape}"
        )
    if future_actions.shape != (48, effective_dimensions):
        raise ValueError(
            "future_actions must have shape "
            f"{(48, effective_dimensions)}, got {future_actions.shape}"
        )
    if model_dimensions < effective_dimensions:
        raise ValueError("model dimensions cannot be smaller than physical dimensions")

    mixed_action = future_actions.copy()
    mixed_action[:, delta_mask] -= state[delta_mask]
    normalized_state = _quantile_normalize(
        state[None],
        state_stats,
        effective_dimensions=effective_dimensions,
        output_dimensions=model_dimensions,
        clamp=clamp,
    )
    normalized_action = _quantile_normalize(
        mixed_action,
        action_stats,
        effective_dimensions=effective_dimensions,
        output_dimensions=model_dimensions,
        clamp=clamp,
    )
    return normalized_state, normalized_action


def make_additive_action_counterfactual(
    future_actions: np.ndarray,
    *,
    action_dimension: int,
    additive_offset: float,
) -> np.ndarray:
    future_actions = np.asarray(future_actions, dtype=np.float32)
    if future_actions.ndim != 2:
        raise ValueError("future_actions must have shape [T, D]")
    if not 0 <= action_dimension < future_actions.shape[1]:
        raise ValueError("action dimension is outside the physical action vector")
    if not np.isfinite(additive_offset):
        raise ValueError("additive action offset must be finite")
    perturbed = future_actions.copy()
    perturbed[:, action_dimension] += additive_offset
    return perturbed


def rgb_frames_to_vae_video(frames: Sequence[np.ndarray]) -> Any:
    import torch

    if not frames:
        raise ValueError("at least one RGB frame is required")
    stacked = np.stack([_validate_rgb(frame) for frame in frames], axis=0)
    video = torch.from_numpy(stacked).permute(3, 0, 1, 2).contiguous()
    return video.unsqueeze(0).to(dtype=torch.float32) / 127.5 - 1.0
