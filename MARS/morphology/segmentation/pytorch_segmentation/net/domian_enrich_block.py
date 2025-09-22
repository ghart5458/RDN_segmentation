import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """Create 3x3 convolution layer with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution (default: 1)
        groups (int): Number of groups for grouped convolution (default: 1)
        dilation (int): Dilation rate for dilated convolution (default: 1)

    Returns:
        nn.Conv2d: 3x3 convolutional layer

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """Create 1x1 convolution layer for channel dimension changes.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution (default: 1)

    Returns:
        nn.Conv2d: 1x1 convolutional layer

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-style architectures.

    Implements a basic residual block with two 3x3 convolutions, batch normalization,
    and ReLU activations with skip connection.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """Initialize BasicBlock.

        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution (default: 1)
            downsample (nn.Module): Downsample layer for skip connection (default: None)
            groups (int): Number of groups for grouped convolution (default: 1)
            base_width (int): Base width for bottleneck (default: 64)
            dilation (int): Dilation rate (default: 1)
            norm_layer (nn.Module): Normalization layer class (default: BatchNorm2d)

        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass through BasicBlock.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after residual block processing

        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DomainEnrich_Block(nn.Module):
    """Domain enrichment block using cascaded BasicBlocks.

    Applies domain-specific feature enrichment through sequential
    residual processing to improve segmentation performance.
    """

    def __init__(self, n_channels, n_classes):
        """Initialize DomainEnrich_Block.

        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output channels/classes

        """
        super().__init__()
        self.basic_block1 = BasicBlock(n_channels,n_classes)
        self.basic_block2 = BasicBlock(n_classes,n_classes)

    def forward(self, x):
        """Forward pass through DomainEnrich_Block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Domain-enriched feature tensor

        """
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        return x

class RDN_Block(nn.Module):
    """Residual Dense Network block with dual domain enrichment paths.

    Implements parallel domain enrichment processing followed by
    feature concatenation for enhanced representation learning.
    """

    def __init__(self, n_channels, n_classes):
        """Initialize RDN_Block.

        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output channels/classes

        """
        super().__init__()
        self.rdn1 = DomainEnrich_Block(n_channels,n_classes)
        self.rdn2 = DomainEnrich_Block(n_channels, n_classes)

    def forward(self, x):
        """Forward pass through RDN_Block.

        Processes input through parallel domain enrichment paths and
        concatenates results for enhanced feature representation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Concatenated features from dual RDN paths

        """
        self.x_rdn1 = self.rdn1(x)
        self.x_rdn2 = self.rdn2(x)
        x = torch.cat((self.x_rdn2, self.x_rdn1), 1)
        # self.x_rdn2 = self.rdn2(self.x_rdn1)
        return x
