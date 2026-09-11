import os
import random
import torch
from toolkit.utils import Image
from .dataset import BaseImageDataset
import torchvision.transforms as transforms

class ImageEnhanceDataset(BaseImageDataset):
    def __init__(self, data_dir, input_size=512, channels=3):
        super().__init__(augment=False)
        self.input_size = input_size
        self.channels = channels
        
        if self.channels == 4:
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        else:
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.to_tensor = transforms.ToTensor()

        self.paths = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        scale_factor = random.randint(2, min(8, self.input_size // 4))  # Prevent zero-size output
        img_obj = Image(self.paths[idx])
        
        if hasattr(img_obj, "convert"):
            # Preserve RGBA if image itself has transparency, even if channels=3 is requested
            if self.channels == 4 or getattr(img_obj, "mode", "") == "RGBA":
                img_obj = img_obj.convert("RGBA")
                self.channels = ensure_channels_match_4 = 4 # Handle dynamically if needed
            else:
                img_obj = img_obj.convert("RGB")

        chunks = img_obj.slice_image(self.input_size)
        chunk = random.choice(chunks)

        t_tgt_raw = self.to_tensor(chunk)
        _, h, w = t_tgt_raw.shape

        # Ensure target sizes are at least 1
        low_h = max(1, h // scale_factor)
        low_w = max(1, w // scale_factor)

        interp_mode = random.choice(["nearest", "bilinear", "bicubic"])
        align_args = {"align_corners": False} if interp_mode in ["bilinear", "bicubic"] else {}

        low = torch.nn.functional.interpolate(
            t_tgt_raw.unsqueeze(0),
            size=(low_h, low_w),
            mode=interp_mode,
            **align_args
        )

        t_inp_raw = torch.nn.functional.interpolate(
            low, size=(h, w), mode="nearest"
        ).squeeze(0)

        t_inp_raw = torch.clamp(t_inp_raw, 0.0, 1.0)

        t_tgt = self.normalize(t_tgt_raw)
        t_inp = self.normalize(t_inp_raw)
        return t_inp, t_tgt
