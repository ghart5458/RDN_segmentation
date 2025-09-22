"""Parts of the U-Net model"""

import torch
import torch.nn.functional as F
from torch import nn


class DomainEnrich(nn.Module):
    """Domain enrichment layer for enhanced feature extraction.

    Applies a convolutional layer followed by batch normalization and ReLU activation
    to enrich domain-specific features in the neural network.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.domain_enrich = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through domain enrichment layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Domain-enriched output tensor

        """
        return self.domain_enrich(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        """Initialize double convolution block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through double convolution block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after double convolution, batch norm, and ReLU

        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """Initialize downsampling block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Forward pass through downsampling block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Downsampled output tensor

        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """Initialize upsampling block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): If True, use bilinear upsampling; otherwise use transpose conv

        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass through upsampling block with skip connection.

        Args:
            x1 (torch.Tensor): Low-resolution feature map to be upsampled
            x2 (torch.Tensor): High-resolution feature map for skip connection

        Returns:
            torch.Tensor: Upsampled and concatenated feature map

        """
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer for final classification.

    Applies a 1x1 convolution to map features to the desired number of output classes.
    """

    def __init__(self, in_channels, out_channels):
        """Initialize output convolution layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes

        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through output convolution.

        Args:
            x (torch.Tensor): Input feature tensor

        Returns:
            torch.Tensor: Class prediction logits

        """
        return self.conv(x)
