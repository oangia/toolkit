import os
import random
import torch
from toolkit.utils import Image
from .dataset import BaseImageDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageEnhanceDataset(BaseImageDataset):
    def __init__(self, data_dir, input_size=512, channels=3):
        super().__init__(augment=False)
        self.input_size = input_size
        self.channels = channels
        
        self.to_tensor = transforms.ToTensor()

        self.paths = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

    def __len__(self):
        return len(self.paths)

    def _normalize_tensor(self, tensor):
        if tensor.shape[0] == 4:
            mean = torch.tensor([0.5, 0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std

    def _augmentations(self, t_inp, t_tgt):
        if self.augment:
            if random.random() > 0.5:
                t_inp = TF.hflip(t_inp)
                t_tgt = TF.hflip(t_tgt)
            if random.random() > 0.5:
                t_inp = TF.vflip(t_inp)
                t_tgt = TF.vflip(t_tgt)
            rot_angle = random.choice([0, 90, 180, 270])
            if rot_angle > 0:
                t_inp = TF.rotate(t_inp, rot_angle)
                t_tgt = TF.rotate(t_tgt, rot_angle)
        return t_inp, t_tgt
        
    def __getitem__(self, idx):
        scale_factor = random.randint(2, min(8, self.input_size // 4))  
        img_obj = Image(self.paths[idx])
        
        if hasattr(img_obj, "convert"):
            if self.channels == 4:
                img_obj = img_obj.convert("RGBA")
            else:
                img_obj = img_obj.convert("RGB")

        chunks = img_obj.slice_image(self.input_size)
        chunk = random.choice(chunks)

        t_tgt_raw = self.to_tensor(chunk)
        
        # Foolproof channel enforcement to override any internal toolkit clipping
        if self.channels == 4:
            if t_tgt_raw.shape[0] == 3:
                alpha = torch.ones((1, t_tgt_raw.shape[1], t_tgt_raw.shape[2]), dtype=t_tgt_raw.dtype, device=t_tgt_raw.device)
                t_tgt_raw = torch.cat([t_tgt_raw, alpha], dim=0)
            elif t_tgt_raw.shape[0] > 4:
                t_tgt_raw = t_tgt_raw[:4, :, :]
        else:
            if t_tgt_raw.shape[0] == 4:
                t_tgt_raw = t_tgt_raw[:3, :, :]

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
        return self._augmentations(t_inp, t_tgt)
