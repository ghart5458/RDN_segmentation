"""ResNet U-Net with Residual Dense Network enhancement.

Combines ResNet backbone with U-Net decoder and RDN domain enrichment
for robust medical image segmentation.
"""

import torch
from torch import nn
from torchvision import models

from .unet_parts import DomainEnrich


def convrelu(in_channels, out_channels, kernel, padding):
    """Create convolution followed by ReLU activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel (int): Kernel size for convolution
        padding (int): Padding for convolution

    Returns:
        nn.Sequential: Conv2d + ReLU sequential module

    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet_RDN(nn.Module):
    """ResNet-based U-Net with RDN domain enrichment.

    Uses pretrained ResNet18 as encoder backbone with U-Net decoder architecture,
    enhanced with dual domain enrichment paths for improved segmentation.
    """

    def __init__(self, n_channels=1, n_classes=3):
        """Initialize ResNetUNet_RDN model.

        Args:
            n_channels (int): Number of input channels (default: 1)
            n_classes (int): Number of output classes (default: 3)

        """
        super().__init__()
        self.rdn1 = DomainEnrich(n_channels, 32)
        self.rdn2 = DomainEnrich(n_channels, 32)

        self.first_conv=convrelu(64,3,1,0)###
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        # self.layer0=self.base_layers[3]
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        # self.layer1 = self.base_layers[4]
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(64, 64, 3, 1)##
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        """Forward pass through ResNetUNet_RDN.

        Processes input through dual RDN paths, then through ResNet encoder
        and U-Net decoder with skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Output segmentation logits

        """
        self.x_rdn1 = self.rdn1(x)
        self.x_rdn2 = self.rdn2(x)

        # self.x_rdn2 = self.rdn2(self.x_rdn1)

        x_original = self.conv_original_size0(torch.cat((self.x_rdn2, self.x_rdn1), 1))
        x_original = self.conv_original_size1(x_original)
        inp=self.first_conv(torch.cat((self.x_rdn2, self.x_rdn1), 1))
        layer0 = self.layer0(inp)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
