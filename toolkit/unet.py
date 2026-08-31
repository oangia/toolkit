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
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512, 512, 512, 512, 512]):
        super().__init__()
        
        encoder_activation = nn.LeakyReLU(0.2, inplace=True)
        decoder_activation = nn.ReLU(inplace=True)
        
        # Encoder (8 stages for 256x256 images)
        self.inc = DoubleConv(in_channels, features[0], activation=encoder_activation)
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
            DoubleConv(features[3], features[4], activation=encoder_activation)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(features[4], features[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[4]),
            encoder_activation,
            DoubleConv(features[4], features[5], activation=encoder_activation)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(features[5], features[5], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[5]),
            encoder_activation,
            DoubleConv(features[5], features[6], activation=encoder_activation)
        )
        self.down7 = nn.Sequential(
            nn.Conv2d(features[6], features[6], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[6]),
            encoder_activation,
            DoubleConv(features[6], features[7], activation=encoder_activation)  # Bottleneck
        )

        # Decoder (8 stages with dropout on the first three upsampling blocks)
        self.up1 = nn.ConvTranspose2d(features[7], features[6], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            DoubleConv(features[6] + features[6], features[6], activation=decoder_activation),
            nn.Dropout(0.5)
        )
        
        self.up2 = nn.ConvTranspose2d(features[6], features[5], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            DoubleConv(features[5] + features[5], features[5], activation=decoder_activation),
            nn.Dropout(0.5)
        )
        
        self.up3 = nn.ConvTranspose2d(features[5], features[4], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            DoubleConv(features[4] + features[4], features[4], activation=decoder_activation),
            nn.Dropout(0.5)
        )
        
        self.up4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(features[3] + features[3], features[3], activation=decoder_activation)
        
        self.up5 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv5 = DoubleConv(features[2] + features[2], features[2], activation=decoder_activation)
        
        self.up6 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv6 = DoubleConv(features[1] + features[1], features[1], activation=decoder_activation)
        
        self.up7 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv7 = DoubleConv(features[0] + features[0], features[0], activation=decoder_activation)

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x = self.up1(x8)
        x = torch.cat([x, x7], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x6], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv5(x)

        x = self.up6(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv6(x)

        x = self.up7(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv7(x)

        return self.tanh(self.outc(x))
