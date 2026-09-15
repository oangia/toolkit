import os
import random
import torch
from toolkit.utils import UImage
from .dataset import BaseImageDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

class ImageEnhanceDataset(BaseImageDataset):
    def __init__(self, data_path, input_size=512, channels=4):
        super().__init__(data_path=data_path, channels=channels)
        self.input_size = input_size
        self.paths = [
            os.path.join(data_path, f)
            for f in sorted(os.listdir(data_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_obj = UImage(self.paths[idx])
        img_obj = img_obj.convert(channels = self.channels)

        chunks = img_obj.slice_image(self.input_size)
        
        target = random.choice(chunks)
        input = self._low(target)

        t_tgt = self.transform(target)
        t_inp = self.transform(input)
        return self._augmentations(t_inp, t_tgt)

    def _low(self, target):
        scale_factor = random.randint(2, min(16, self.input_size // 4))  
        w, h = target.size    
        low_h = max(1, h // scale_factor)
        low_w = max(1, w // scale_factor)
    
        # Map string modes to PIL Resampling filters
        methods_map = {
            "nearest": PILImage.Resampling.NEAREST if hasattr(PILImage, 'Resampling') else PILImage.NEAREST,
            "bilinear": PILImage.Resampling.BILINEAR if hasattr(PILImage, 'Resampling') else PILImage.BILINEAR,
            "bicubic": PILImage.Resampling.BICUBIC if hasattr(PILImage, 'Resampling') else PILImage.BICUBIC,
        }
        
        interp_name = random.choice(list(methods_map.keys()))
        downscale_filter = methods_map[interp_name]
    
        # 1. Downscale the image
        low_img = target.resize((low_w, low_h), resample=downscale_filter)
    
        # 2. Upscale back to original size using nearest neighbor (matching your torch code)
        upscale_filter = PILImage.Resampling.NEAREST if hasattr(PILImage, 'Resampling') else PILImage.NEAREST
        inp_img = low_img.resize((w, h), resample=upscale_filter)
    
        return inp_img

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
        
