from .domian_enrich_block import DomainEnrich_Block, RDN_Block
from .MLP import MLP
from .res_unet_rdn import ResNetUNet_RDN
from .resnet_unet import ResNetUNet
from .unet_light import UNet_Light
from .unet_light_rdn import UNet_Light_RDN
from .unet_model import UNet
from .unet_rdn import UNet_RDN

# UNet_Wavelet needs pytorch_wavelets, which is installed from GitHub rather than
# PyPI (see setup_pytorch_wavelets.py) and is not present in every environment.
# None of the shipped models use it, so a missing install must not take down the
# segmentation and training paths that import this package.
try:
    from .unet_wavelet import UNet_Wavelet
except ImportError:  # pragma: no cover - depends on optional dependency
    UNet_Wavelet = None
