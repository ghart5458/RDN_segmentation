"""Full assembly of the parts to form the complete network"""

import torch
from torch import nn

from .domian_enrich_block import DomainEnrich_Block
from .unet_parts import DoubleConv, Down, OutConv, Up


class UNet_Light_RDN(nn.Module):
    """Lightweight U-Net with Residual Dense Network (RDN) blocks.

    This model combines the U-Net architecture with RDN blocks for enhanced
    feature extraction. The RDN blocks utilize residual connections and dense
    connections to better capture hierarchical features for medical image segmentation.

    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output segmentation classes
        bilinear (bool): If True, use bilinear upsampling; otherwise use transpose convolutions

    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # RDN feature extraction blocks
        self.rdn1 = DomainEnrich_Block(n_channels, 16)
        self.rdn2 = DomainEnrich_Block(n_channels, 16)

        # U-Net encoder
        self.inc = DoubleConv(32, 32)  # Takes concatenated RDN outputs
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        # U-Net decoder
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)

        # Output classification layer
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        """Forward pass through UNet_Light_RDN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes, height, width)

        """
        # Extract domain-enriched features using RDN blocks
        self.x_rdn1 = self.rdn1(x)
        self.x_rdn2 = self.rdn2(x)

        # Combine RDN features and pass through U-Net encoder
        x1 = self.inc(torch.cat((self.x_rdn2, self.x_rdn1), 1))

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
