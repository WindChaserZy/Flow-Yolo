import torch
from torch import nn

class FeatureEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(FeatureEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x : torch.Tensor):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        return x