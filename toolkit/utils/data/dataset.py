import os
import random
import torch
import cv2
import numpy as np
from toolkit.utils import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class BaseImageDataset(Dataset):
    def __init__(self, inputs = None, targets = None, augment=False):
        self.inputs = inputs
        self.targets = targets
        self.augment = augment
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def loader(self, batch_size=2, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

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

    def _prepare_image(self, img):
        if hasattr(img, "detach"):
            img = img.detach().cpu()
        
        img = torch.clamp((img * 0.5) + 0.5, 0, 1)
        if hasattr(img, "numpy"):
            img = img.numpy()
            
        if img.ndim == 3 and img.shape[0] in [3, 4]:
            img = np.transpose(img, (1, 2, 0))
            
        return img
    
    def _inp(self, t_inp):
        return self._prepare_image(t_inp)
        
    def _out(self, t_inp, t_tgt):
        return self._prepare_image(t_tgt)
        
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
