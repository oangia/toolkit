import os
import random
import torch
import cv2
import numpy as np
from toolkit import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class BaseImageDataset(Dataset):
    def __init__(self, inputs = None, targets = None, augment=False):
        self.augment = augment
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales [0, 1] to [-1, 1]
        ])

    def __len__(self):
        return len(self.inputs)

    def _augmentations(self, t_inp, t_tgt):
        if self.augment:
            if random.random() > 0.5:
                t_inp = TF.hflip(t_inp)
                t_tgt = TF.hflip(t_tgt)
            if random.random() > 0.5:
                t_inp = TF.vflip(t_inp)
                t_tgt = TF.vflip(t_tgt)
            rot_angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270])
            if rot_angle > 0:
                t_inp = TF.rotate(t_inp, rot_angle)
                t_tgt = TF.rotate(t_tgt, rot_angle)
        return t_inp, t_tgt

    def __getitem__(self, idx):
        t_inp = self.inputs[idx]
        t_tgt = self.targets[idx]
        return self._augmentations(t_inp, t_tgt)

    def _inp(self, t_inp):
        return torch.clamp((t_inp * 0.5) + 0.5, 0, 1).permute(1, 2, 0).numpy()
        
    def _out(self, t_inp, t_tgt):
        return torch.clamp((t_tgt * 0.5) + 0.5, 0, 1).permute(1, 2, 0).numpy()
        
    def show_sample(self, limit=None):
        total = len(self)
        num_to_show = total if limit is None else min(total, limit)
        for idx in range(num_to_show):
            t_inp, t_tgt = self[idx]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(self._inp(t_inp))
            axes[0].set_title(f"Input [{idx}]")
            axes[0].axis('off')

            axes[1].imshow(self._out(t_inp, t_tgt))
            axes[1].set_title(f"Target [{idx}]")
            axes[1].axis('off')

            plt.show()
