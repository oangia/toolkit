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
        input = self._low(UImage(target).lower_quality().toPIL())

        t_tgt = self.transform(target)
        t_inp = self.transform(input)
        return self._augmentations(t_inp, t_tgt)

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
        
