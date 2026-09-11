import os
import random
import torch
import cv2
import numpy as np
from toolkit.utils import Image, BaseImageDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
        
class MultiPairDataset(BaseImageDataset):
    def __init__(self, inp_paths, tgt_paths, augment=False, input_size=256):
        super().__init__(augment=augment)
        self.inputs = []
        self.targets = []
        self.augment = augment
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales [0, 1] to [-1, 1]
        ])

        for inp_p, tgt_p in zip(inp_paths, tgt_paths):
            inp_img = Image(inp_p).resize(input_size, input_size, keep_aspect_ratio=False).slice_image(input_size)
            tgt_img = Image(tgt_p).resize(input_size, input_size, keep_aspect_ratio=False).slice_image(input_size)
            for i_inp, i_tgt in zip(inp_img, tgt_img):
                t_inp = transform(i_inp)
                t_tgt = transform(i_tgt)

                if torch.equal(t_inp, t_tgt):
                    continue

                self.inputs.append(t_inp)
                self.targets.append(t_tgt)

class ImageDataset(BaseImageDataset):
    def __init__(self, folder_path, input_files, target_files, length=None, augment=False, input_size=256):
        super().__init__(augment=augment)
        self.inputs = []
        self.targets = []
        self.augment = augment
        
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales [0, 1] to [-1, 1]
        ])

        for inp_f, tgt_f in zip(input_files, target_files):
            inp_p = os.path.join(folder_path, inp_f)
            tgt_p = os.path.join(folder_path, tgt_f)

            if not os.path.exists(inp_p) or not os.path.exists(tgt_p):
                print(f"Warning: Missing file pair -> {inp_f} or {tgt_f}")
                continue

            inp_img = Image(inp_p).resize(input_size, input_size, keep_aspect_ratio=False).slice_image(input_size)
            tgt_img = Image(tgt_p).resize(input_size, input_size, keep_aspect_ratio=False).slice_image(input_size)
            
            for i_inp, i_tgt in zip(inp_img, tgt_img):
                t_inp = transform(i_inp)
                t_tgt = transform(i_tgt)

                if torch.equal(t_inp, t_tgt):
                    continue

                self.inputs.append(t_inp)
                self.targets.append(t_tgt)
