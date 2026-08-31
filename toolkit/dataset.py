import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from toolkit import Image

class MultiPairDataset(Dataset):
    def __init__(self, inp_paths, tgt_paths, length=None, augment=False, input_size=256):
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

        self.length = length
        if length is None:
            self.length = len(self.inputs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = idx % len(self.inputs)
        t_inp = self.inputs[real_idx]
        t_tgt = self.targets[real_idx]

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
