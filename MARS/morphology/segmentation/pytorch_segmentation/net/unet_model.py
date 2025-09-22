"""Full assembly of the parts to form the complete network"""

from torch import nn

from .unet_parts import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """Standard U-Net implementation for medical image segmentation.

    The U-Net architecture consists of an encoder-decoder structure with skip connections.
    The encoder progressively downsamples the input, while the decoder upsamples and
    combines features from corresponding encoder levels via skip connections.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        n_classes (int): Number of output classes for segmentation
        bilinear (bool): If True, use bilinear upsampling; otherwise use transpose convolutions

    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoder path
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        """Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes, height, width)

        """
        x1 = self.inc(x)
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
