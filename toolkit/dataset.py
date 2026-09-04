import os
import random
import torch
import cv2
import numpy as np
from toolkit import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import toolkit.nn as inn
        
class MultiPairDataset(inn.BaseImageDataset):
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

class ImageEnhanceDataset(inn.BaseImageDataset):
    def __init__(self, data_dir, input_size=256, scale_factor=8):
        super().__init__(augment=False)
        self.inputs = []
        self.targets = []
        
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        paths = [
            os.path.join(data_dir, f) 
            for f in sorted(os.listdir(data_dir)) 
            if f.lower().endswith(valid_extensions)
        ]

        for path in paths:
            chunks = Image(path).slice_image(input_size)
            for chunk in chunks:
                # 1. Convert to tensor in [0, 1] range first
                t_tgt_raw = to_tensor(chunk)
                _, h, w = t_tgt_raw.shape
                
                # 2. Downsize then upscale in [0, 1] space
                low = torch.nn.functional.interpolate(
                    t_tgt_raw.unsqueeze(0), 
                    size=(h // scale_factor, w // scale_factor), 
                    mode='area'
                )

                t_inp_raw = torch.nn.functional.interpolate(
                    low, 
                    size=(h, w), 
                    mode='bicubic', 
                    align_corners=False
                ).squeeze(0)

                # 3. Clamp input to prevent bicubic overshooting artifacts
                t_inp_raw = torch.clamp(t_inp_raw, 0.0, 1.0)
                
                # 4. Normalize both to [-1, 1] for the model
                self.targets.append(normalize(t_tgt_raw))
                self.inputs.append(normalize(t_inp_raw))

class ImageDataset(inn.BaseImageDataset):
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

class YOLODataset(inn.BaseImageDataset):
    def __init__(self, data_dir, img_size=256, augment=True):
        super().__init__(augment=augment)
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.img_size = img_size
        self.inputs = self.image_filenames

        matched_labels = 0
        for img_name in self.image_filenames:
            label_name = os.path.splitext(img_name)[0] + '.lbl'
            if os.path.exists(os.path.join(self.label_dir, label_name)):
                matched_labels += 1
        print(f"Found {matched_labels} matching label files out of {len(self.image_filenames)} images.")

    def _transform_box_arbitrary(self, cx, cy, w, h, angle, flip_h, img_w, img_h):
        xmin = (cx - w / 2) * img_w
        xmax = (cx + w / 2) * img_w
        ymin = (cy - h / 2) * img_h
        ymax = (cy + h / 2) * img_h

        corners = np.array([
            [xmin, ymin, 1],
            [xmax, ymin, 1],
            [xmax, ymax, 1],
            [xmin, ymax, 1]
        ])

        center = (img_w / 2.0, img_h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed_corners = np.dot(M, corners.T).T

        if flip_h:
            transformed_corners[:, 0] = img_w - transformed_corners[:, 0]

        new_xmin = np.clip(np.min(transformed_corners[:, 0]), 0, img_w)
        new_xmax = np.clip(np.max(transformed_corners[:, 0]), 0, img_w)
        new_ymin = np.clip(np.min(transformed_corners[:, 1]), 0, img_h)
        new_ymax = np.clip(np.max(transformed_corners[:, 1]), 0, img_h)

        if new_xmax <= new_xmin or new_ymax <= new_ymin:
            return None

        new_cx = ((new_xmin + new_xmax) / 2.0) / img_w
        new_cy = ((new_ymin + new_ymax) / 2.0) / img_h
        new_w = (new_xmax - new_xmin) / img_w
        new_h = (new_ymax - new_ymin) / img_h

        if new_w <= 0 or new_h <= 0:
            return None

        return new_cx, new_cy, new_w, new_h

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.img_size, self.img_size))

        angle = 0.0
        flip_h = False

        if self.augment:
            angle_steps = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            angle = float(random.choice(angle_steps))
            flip_h = random.choice([True, False])

            center = (self.img_size / 2.0, self.img_size / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (self.img_size, self.img_size),
                                   borderMode=cv2.BORDER_REFLECT_101)

            if flip_h:
                image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Scale inputs to [-1, 1] range
        image = image * 2.0 - 1.0

        target_small = torch.zeros(32, 32, 5)
        target_medium = torch.zeros(16, 16, 5)
        target_large = torch.zeros(8, 8, 5)

        label_name = os.path.splitext(img_name)[0] + '.lbl'
        label_path = os.path.join(self.label_dir, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, cx, cy, w, h = map(float, parts)

                        transformed = self._transform_box_arbitrary(cx, cy, w, h, angle, flip_h, self.img_size, self.img_size)
                        if transformed is None:
                            continue
                        cx, cy, w, h = transformed

                        if max(w, h) < 0.25:
                            gx, gy = min(int(cx * 32), 31), min(int(cy * 32), 31)
                            target_small[gy, gx] = torch.tensor([1.0, (cx * 32) - gx, (cy * 32) - gy, w, h])
                        elif max(w, h) < 0.55:
                            gx, gy = min(int(cx * 16), 15), min(int(cy * 16), 15)
                            target_medium[gy, gx] = torch.tensor([1.0, (cx * 16) - gx, (cy * 16) - gy, w, h])
                        else:
                            gx, gy = min(int(cx * 8), 7), min(int(cy * 8), 7)
                            target_large[gy, gx] = torch.tensor([1.0, (cx * 8) - gx, (cy * 8) - gy, w, h])

        return image, (target_small, target_medium, target_large)

    def _out(self, t_inp, t_tgt):
        target_small, target_medium, target_large = t_tgt
        img = (self._inp(t_inp) * 255).astype(np.uint8).copy()
        
        grids = [
            (target_small, 32, (255, 0, 0)),    # Small scale grid (Blue lines)
            (target_medium, 16, (0, 255, 255)), # Medium scale grid (Yellow lines)
            (target_large, 8, (255, 0, 255))    # Large scale grid (Magenta lines)
        ]
        
        # 1. Draw grid cell lines over the image to visualize structural resolution
        for _, S, color in grids:
            cell_size = self.img_size // S
            for i in range(1, S):
                pt = i * cell_size
                # Draw faint grid lines
                cv2.line(img, (pt, 0), (pt, self.img_size), color, 1)
                cv2.line(img, (0, pt), (self.img_size, pt), color, 1)

        # 2. Draw object bounding boxes mapped to the feature grid
        for grid, S, _ in grids:
            for gy in range(S):
                for gx in range(S):
                    if grid[gy, gx, 0] > 0:
                        cell_data = grid[gy, gx].tolist()
                        _, tx, ty, w, h = cell_data
                        
                        cx = (gx + tx) / S
                        cy = (gy + ty) / S
                        
                        xmin = int((cx - w / 2) * self.img_size)
                        xmax = int((cx + w / 2) * self.img_size)
                        ymin = int((cy - h / 2) * self.img_size)
                        ymax = int((cy + h / 2) * self.img_size)
                        
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        
        return img.astype(np.float32) / 255.0
