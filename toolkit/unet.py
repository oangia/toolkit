import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 256)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 1024))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 1024))  # Bottleneck

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024 + 1024, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(1024 + 1024, 1024)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512 + 512, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256 + 256, 256)

        self.outc = nn.Conv2d(256, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        return self.sigmoid(self.outc(x))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):  # Takes input image + target/output image concatenated (6 channels)
        super().__init__()
        self.net = nn.Sequential(
            # Input: 256x256x6
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final output map of patches
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        # Concatenate input and target/output along the channel dimension
        cat_in = torch.cat([x, y], dim=1)
        return self.net(cat_in)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels for X and Y gradients (fixed torch.float32 typo)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Register as buffers so they move to GPU with the model
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1)) # For 3 channels (RGB)
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))
        self.criterion = nn.L1Loss()

    def get_edges(self, img):
        # Apply convolution to get X and Y gradients
        # groups=3 ensures each RGB channel is filtered independently
        edge_x = torch.nn.functional.conv2d(img, self.sobel_x, padding=1, groups=3)
        edge_y = torch.nn.functional.conv2d(img, self.sobel_y, padding=1, groups=3)
        # Magnitude of edges
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

    def forward(self, fake, target):
        fake_edges = self.get_edges(fake)
        target_edges = self.get_edges(target)
        return self.criterion(fake_edges, target_edges)
# ==========================================
# DATASET
# ==========================================
class MultiPairDataset(Dataset):
    def __init__(self, inp_paths, tgt_paths):
        self.inputs = []
        self.targets = []
        transform = transforms.ToTensor()

        for inp_p, tgt_p in zip(inp_paths, tgt_paths):
            inp_img = Image.open(inp_p).convert("RGB").resize(input_size)
            tgt_img = Image.open(tgt_p).convert("RGB").resize(input_size)
            self.inputs.append(transform(inp_img))
            self.targets.append(transform(tgt_img))

    def __len__(self):
        return 2 #len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]