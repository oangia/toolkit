import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.double_conv(x)
        
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[128, 256, 512, 1024, 1024]):
        super().__init__()
        
        encoder_activation = nn.LeakyReLU(0.2, inplace=True)
        decoder_activation = nn.ReLU(inplace=True)
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0], activation=encoder_activation)
        
        # Replaced MaxPool2d with strided convolutions for Pix2Pix downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            encoder_activation,
            DoubleConv(features[0], features[1], activation=encoder_activation)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(features[1], features[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            encoder_activation,
            DoubleConv(features[1], features[2], activation=encoder_activation)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(features[2], features[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            encoder_activation,
            DoubleConv(features[2], features[3], activation=encoder_activation)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(features[3], features[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            encoder_activation,
            DoubleConv(features[3], features[4], activation=encoder_activation)  # Bottleneck
        )

        # Decoder (Added Dropout to upper layers like Pix2Pix)
        self.up1 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            DoubleConv(features[3] + features[3], features[3], activation=decoder_activation),
            nn.Dropout(0.5)
        )
        
        self.up2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            DoubleConv(features[2] + features[2], features[2], activation=decoder_activation),
            nn.Dropout(0.5)
        )
        
        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(features[1] + features[1], features[1], activation=decoder_activation)
        
        self.up4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(features[0] + features[0], features[0], activation=decoder_activation)

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Pix2Pix uses Tanh (requires images normalized to [-1, 1])

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

        return self.tanh(self.outc(x))
