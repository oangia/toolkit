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
        
        # REMOVED self.normalize entirely to stop it from crashing on 4 channels
        self.to_tensor = transforms.ToTensor()

        self.paths = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

    def __len__(self):
        return len(self.paths)

    def _normalize_tensor(self, tensor):
        # Force dynamic calculation based on actual tensor channels at runtime
        if tensor.shape[0] == 4:
            mean = torch.tensor([0.5, 0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, idx):
        scale_factor = random.randint(2, min(8, self.input_size // 4))  
        img_obj = Image(self.paths[idx])
        
        if hasattr(img_obj, "convert"):
            # Uniformly force the conversion to match self.channels globally
            if self.channels == 4:
                img_obj = img_obj.convert("RGBA")
            else:
                img_obj = img_obj.convert("RGB")

        chunks = img_obj.slice_image(self.input_size)
        chunk = random.choice(chunks)

        t_tgt_raw = self.to_tensor(chunk)
        _, h, w = t_tgt_raw.shape

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

        t_tgt = self._normalize_tensor(t_tgt_raw)
        t_inp = self._normalize_tensor(t_inp_raw)
        return t_inp, t_tgt
