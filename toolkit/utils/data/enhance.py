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
        scale_factor = random.randint(2, 16)
        img_obj = Image(self.paths[idx])
        
        if hasattr(img_obj, "convert"):
            if self.channels == 4:
                img_obj = img_obj.convert("RGBA")
            else:
                img_obj = img_obj.convert("RGB")

        chunks = img_obj.slice_image(self.input_size)
        chunk = random.choice(chunks)

        t_tgt_raw = self.to_tensor(chunk)
        _, h, w = t_tgt_raw.shape

        low = torch.nn.functional.interpolate(
            t_tgt_raw.unsqueeze(0),
            size=(h // scale_factor, w // scale_factor),
            mode=random.choice(["nearest", "bilinear", "bicubic"]),
        )

        t_inp_raw = torch.nn.functional.interpolate(
            low, size=(h, w), mode="nearest"
        ).squeeze(0)

        t_inp_raw = torch.clamp(t_inp_raw, 0.0, 1.0)

        t_tgt = self.normalize(t_tgt_raw)
        t_inp = self.normalize(t_inp_raw)
        return t_inp, t_tgt
