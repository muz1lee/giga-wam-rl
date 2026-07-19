from __future__ import annotations

import importlib.util
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from giga_wam_rl.gwp05_pilot_counterfactual import (
    _encode_prompt,
    _load_transformer,
    _load_vae,
)
from giga_wam_rl.gwp05_pilot_inputs import compose_t_layout
from giga_wam_rl.gwp05_smoke import _add_upstream_import_paths
from giga_wam_rl.robotwin_collection import (
    ACTION_HORIZON,
    MODEL_ACTION_DIM,
    PolicyPrediction,
    PolicyActionNormalizer,
)


class GWP05ActionPolicy:
    def __init__(
        self,
        *,
        pipeline: Any,
        normalizer: PolicyActionNormalizer,
        prompt_embedding: Any,
        device: Any,
        dtype: Any,
        num_inference_steps: int,
        clip_normalized_actions: bool,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num inference steps must be positive")
        self.pipeline = pipeline
        self.normalizer = normalizer
        self.prompt_embedding = prompt_embedding
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = int(num_inference_steps)
        self.clip_normalized_actions = bool(clip_normalized_actions)

    def predict(
        self,
        *,
        cameras: dict[str, np.ndarray],
        state: np.ndarray,
        seed: int,
    ) -> PolicyPrediction:
        import torch

        composite = compose_t_layout(
            cameras["cam_high"],
            cameras["cam_left_wrist"],
            cameras["cam_right_wrist"],
        )
        reference_image = Image.fromarray(composite)
        normalized_state = torch.from_numpy(self.normalizer.normalize_state(state)).to(
            device=self.device, dtype=self.dtype
        )
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        normalized_action = self.pipeline(
            height=384,
            width=320,
            action_chunk=ACTION_HORIZON,
            state=normalized_state,
            num_frames=5,
            guidance_scale=0.0,
            num_inference_steps=self.num_inference_steps,
            image=reference_image,
            return_dict=False,
            prompt_embeds=self.prompt_embedding,
            action_dim=MODEL_ACTION_DIM,
            generator=generator,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        if isinstance(normalized_action, tuple):
            normalized_action = normalized_action[0]
        normalized_array = normalized_action.detach().float().cpu().numpy()
        if normalized_array.shape == (1, ACTION_HORIZON, MODEL_ACTION_DIM):
            normalized_array = normalized_array[0]
        if normalized_array.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
            raise ValueError(
                "GWP action output must have shape [48,16], got "
                f"{normalized_array.shape}"
            )
        if not np.isfinite(normalized_array).all():
            raise ValueError("GWP action output contains non-finite values")
        physical_action = self.normalizer.denormalize_action(
            normalized_array,
            state,
            clip_normalized=self.clip_normalized_actions,
        )
        return PolicyPrediction(
            normalized_action=normalized_array.astype(np.float32, copy=True),
            physical_action=physical_action,
            inference_time_s=float(elapsed),
            seed=int(seed),
        )


def _load_upstream_inference_module(upstream_root: Path) -> Any:
    _add_upstream_import_paths(upstream_root)
    script_path = upstream_root / "scripts" / "inference_openloop.py"
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    spec = importlib.util.spec_from_file_location(
        "giga_wam_rl._pinned_gwp05_inference", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned GWP inference module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gwp05_action_policy(
    *,
    checkpoint: Path,
    base_model: Path,
    upstream_root: Path,
    norm_stats_path: Path,
    prompt: str,
    device_name: str,
    num_inference_steps: int = 10,
    clip_normalized_actions: bool = True,
    compile_transformer: bool = False,
) -> GWP05ActionPolicy:
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GWP RoboTwin collection requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    prompt_embedding = _encode_prompt(base_model, prompt, device=device, dtype=dtype)
    vae = _load_vae(base_model, device=device, dtype=dtype)
    transformer = _load_transformer(
        checkpoint, upstream_root, device=device, dtype=dtype
    )
    upstream_inference = _load_upstream_inference_module(upstream_root)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipeline = upstream_inference.WAPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    ).to(device=device, dtype=dtype)
    if compile_transformer:
        compiled = upstream_inference.compile_policy_action_blocks(
            pipeline,
            mode="reduce-overhead",
            fullgraph=False,
            scope="action-blocks",
        )
        if not compiled:
            raise RuntimeError(
                "torch.compile requested but no GWP action blocks compiled"
            )
    return GWP05ActionPolicy(
        pipeline=pipeline,
        normalizer=PolicyActionNormalizer.from_json(norm_stats_path),
        prompt_embedding=prompt_embedding,
        device=device,
        dtype=dtype,
        num_inference_steps=num_inference_steps,
        clip_normalized_actions=clip_normalized_actions,
    )
