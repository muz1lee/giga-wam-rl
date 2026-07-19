import functools

import torch
import torch.nn.functional as torch_f
from world_action_model.trainer.wa_casual_trainer_mot import CasualWATrainerMoT

from giga_wam_rl import deterministic_transform as _deterministic_transform  # noqa: F401
from giga_wam_rl import valid_window_dataset as _valid_window_dataset  # noqa: F401
from giga_wam_rl.future_flow import (
    build_clean_action_timestep,
    future_flow_batch,
)


class FailureFutureTrainerMoT(CasualWATrainerMoT):
    """Post-train future dynamics while conditioning on clean physical actions."""

    def forward_step(self, batch_dict):
        if not self.expand_timesteps:
            raise ValueError("FailureFutureTrainerMoT requires expand_timesteps=True")
        if "ref_images" not in batch_dict:
            raise ValueError("FailureFutureTrainerMoT requires ref_images")

        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        action = batch_dict["action"]
        state = batch_dict["state"]
        batch_size = images.shape[0]
        visual_timestep, visual_sigma = self.get_timestep_and_sigma(
            batch_size, images.ndim, self.visual_flow_shift
        )

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        future_noise = torch.randn_like(visual_latents[:, :, 1:])
        future_latents, noisy_future, future_target = future_flow_batch(
            visual_latents,
            sigma=visual_sigma,
            noise=future_noise,
        )
        ref_latents = self.forward_vae(batch_dict["ref_images"][:, :1])
        if ref_latents.shape[2] != 1:
            raise ValueError("reference VAE encoding must contain one latent frame")

        tokens_per_frame = future_latents.shape[-2] * future_latents.shape[-1] // 4
        timestep = build_clean_action_timestep(
            visual_timestep,
            num_state_tokens=state.shape[1],
            num_ref_tokens=tokens_per_frame * ref_latents.shape[2],
            num_action_tokens=action.shape[1],
            num_future_tokens=tokens_per_frame * future_latents.shape[2],
            dtype=self.dtype,
            device=noisy_future.device,
        )

        visual_pred, _ = transformer(
            ref_latents=ref_latents.to(self.dtype),
            noisy_latents=noisy_future.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=batch_dict["prompt_embeds"].to(self.dtype),
            return_dict=False,
            action=action.to(self.dtype),
            state=state.to(self.dtype),
            embodiment_id=batch_dict["embodiment_id"],
        )
        future_pred = visual_pred[:, :, 1:]
        if future_pred.shape != future_target.shape:
            raise ValueError(
                "future prediction/target shapes differ: "
                f"{future_pred.shape} != {future_target.shape}"
            )
        return {
            "future_visual_loss": torch_f.mse_loss(
                future_pred.float(), future_target.float()
            )
        }
