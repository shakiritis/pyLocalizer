import torch.nn as nn

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=(1, 1), p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CNN2D_TimeAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock2D(3, 32, k=3, s=(1, 1), p=1),
            ConvBlock2D(32, 64, k=3, s=(2, 1), p=1),
        )
        self.mid = nn.Sequential(
            ConvBlock2D(64, 128, k=3, s=(2, 1), p=1),
            ConvBlock2D(128, 128, k=3, s=(1, 1), p=1),
            ConvBlock2D(128, 256, k=3, s=(2, 1), p=1),
            ConvBlock2D(256, 256, k=3, s=(1, 1), p=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mid(x)
        return self.head(x)
