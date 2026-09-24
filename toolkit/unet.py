import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Expands channels by 4x, then rearranges them to double height & width
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.up(x)
        
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[128, 256, 256, 512]):
        super().__init__()
        
        encoder_activation = nn.LeakyReLU(0.2, inplace=True)
        decoder_activation = nn.ReLU(inplace=True)
        
        # Encoder (3 stages for 256x256 images -> Bottleneck is 32x32)
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
            DoubleConv(features[2], features[3], activation=encoder_activation)  # Bottleneck
        )

        # Decoder (3 stages with light 0.1 dropout on the first upsampling block)
        self.up1 = PixelShuffleUp(features[3], features[2])
        self.conv1 = nn.Sequential(
            DoubleConv(features[2] + features[2], features[2], activation=decoder_activation),
            nn.Dropout(0.1)
        )
        
        self.up2 = PixelShuffleUp(features[2], features[1])
        self.conv2 = DoubleConv(features[1] + features[1], features[1], activation=decoder_activation)
        
        self.up3 = PixelShuffleUp(features[1], features[0])
        self.conv3 = DoubleConv(features[0] + features[0], features[0], activation=decoder_activation)

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # Bottleneck ($32 \times 32$)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.tanh(self.outc(x))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6): # 3 channels for input + 3 channels for target/fake
        super(PatchDiscriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 1, 4, padding=1) # Outputs a patch map of logits
        )

    def forward(self, img_input, img_target):
        # Concatenate image and condition/target along channels
        img_input = torch.cat((img_input, img_target), 1)
        return self.model(img_input)

# --- Initialization Script ---
# Instantiate train and validation sets separately
# train_dataset = ImageDataset(folder_path=drive_path, input_files=train_inputs, target_files=train_targets, length=100, augment=True, input_size=input_size)
# val_dataset = ImageDataset(folder_path=drive_path, input_files=val_inputs, target_files=val_targets, augment=False, input_size=input_size)
