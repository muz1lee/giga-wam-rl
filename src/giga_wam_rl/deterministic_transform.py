import torch
from giga_train import TRANSFORMS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f
from world_action_model.transforms.wa_transforms_lerobot_pretrain import (
    WALeRobotTransformsPretrain,
)


@TRANSFORMS.register
class DeterministicWALeRobotTransforms(WALeRobotTransformsPretrain):
    """Use a center crop so a fixed failure window has fixed visual inputs."""

    def _process_images(
        self, input_images: torch.Tensor, dst_width: int, dst_height: int
    ) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        height = int(input_images.shape[2])
        width = int(input_images.shape[3])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        input_images = vision_f.resize(
            input_images,
            (new_height, new_width),
            InterpolationMode.BILINEAR,
        )
        crop_left = (new_width - dst_width) // 2
        crop_top = (new_height - dst_height) // 2
        input_images = vision_f.crop(
            input_images,
            crop_top,
            crop_left,
            dst_height,
            dst_width,
        )
        return self.normalize(input_images)
