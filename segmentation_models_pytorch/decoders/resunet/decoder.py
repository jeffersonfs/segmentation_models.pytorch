import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from .modules import ResidualConv, Upsample


class ResUnetDecoder(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        # (320, 112, 40, 24, 16)
        super(ResUnetDecoder, self).__init__()

        self.bridge = ResidualConv(filters[4], filters[2], 2, 1)

        self.upsample_1 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 4, 4)
        self.up_residual_conv2 = ResidualConv(80, 80, 1, 1)

        self.upsample_3 = Upsample(80, 80, 4, 4)
        self.up_residual_conv3 = ResidualConv(112, 16, 1, 1)

        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
        )

    def forward(self, *features):
        # Encode
        x1 = features[1] # 16
        x2 = features[3] # 40 
        x3 = features[5] # 320


        # Bridge
        x4 = self.bridge(x3) # 40
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
