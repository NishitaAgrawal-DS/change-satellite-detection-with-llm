import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=5):  # 🔥 IMPORTANT
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)

        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = ConvBlock(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x1, x2):
        s1 = self.enc1(x1)
        f1 = self.enc2(self.pool(s1))

        s2 = self.enc1(x2)
        f2 = self.enc2(self.pool(s2))

        diff = torch.abs(f1 - f2)

        x = self.up(diff)
        x = torch.cat([x, s1], dim=1)
        x = self.dec(x)

        return self.final(x)