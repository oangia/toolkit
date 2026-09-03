import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from toolkit import Image

class BaseImageDataset(Dataset):
    def __init__(self, inputs = None, targets = None, augment=False):
        self.augment = augment

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
        
    def show_sample(self, idx):
        t_inp, t_tgt = self[idx]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(self._inp(t_inp))
        axes[0].set_title("Input")
        axes[0].axis('off')

        axes[1].imshow(self._out(t_inp, t_tgt) )
        axes[1].set_title("Target")
        axes[1].axis('off')

        plt.show()
        
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

class YOLODataset(BaseImageDataset):
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

        if flip_h:
            corners[:, 0] = img_w - corners[:, 0]

        center = (img_w / 2, img_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        transformed_corners = np.dot(M, corners.T).T

        new_xmin = np.clip(np.min(transformed_corners[:, 0]), 0, img_w)
        new_xmax = np.clip(np.max(transformed_corners[:, 0]), 0, img_w)
        new_ymin = np.clip(np.min(transformed_corners[:, 1]), 0, img_h)
        new_ymax = np.clip(np.max(transformed_corners[:, 1]), 0, img_h)

        if new_xmax <= new_xmin or new_ymax <= new_ymin:
            return None

        new_cx = ((new_xmin + new_xmax) / 2) / img_w
        new_cy = ((new_ymin + new_ymax) / 2) / img_h
        new_w = (new_xmax - new_xmin) / img_w
        new_h = (new_ymax - new_ymin) / img_h

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

            center = (self.img_size / 2, self.img_size / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(image, M, (self.img_size, self.img_size),
                                   borderMode=cv2.BORDER_REFLECT_101)

            if flip_h:
                image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

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
        img = (self._inp(t_inp) * 255).astype(np.uint8).copy()
        
        for box in t_tgt:
            cls_id, cx, cy, w, h = box.tolist()
            xmin = int((cx - w / 2) * self.img_size)
            xmax = int((cx + w / 2) * self.img_size)
            ymin = int((cy - h / 2) * self.img_size)
            ymax = int((cy + h / 2) * self.img_size)
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
        return img.astype(np.float32) / 255.0
