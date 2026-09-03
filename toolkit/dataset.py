import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from toolkit import Image

class BaseImageDataset(Dataset):
    def __init__(self, inputs, targets, augment=False):
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def _apply_augmentations(self, t_inp, t_tgt):
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
        return self._apply_augmentations(t_inp, t_tgt)
        
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

class ImageEnhanceDataset(BaseImageDataset):
    def __init__(self, tgt_paths, input_size=256, scale_factor=8):
        super().__init__(augment=False)
        self.inputs = []
        self.targets = []
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for path in tgt_paths:
            chunks = Image(path).slice_image(input_size)
            for chunk in chunks:
                t_tgt = transform(chunk)
                _, h, w = t_tgt.shape
                
                # Downsize then upscale back to create input
                low = torch.nn.functional.interpolate(
                    t_tgt.unsqueeze(0), 
                    size=(h // scale_factor, w // scale_factor), 
                    mode='area'
                )

                # Upscale back to original size using 'bicubic' for smooth expansion
                t_inp = torch.nn.functional.interpolate(
                    low, 
                    size=(h, w), 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze(0)

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
