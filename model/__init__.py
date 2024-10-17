from .deeplabv3p import DeepLabV3Plus
from .light_munet import LightMUNet
from .unet import UNet
from .vmunet import VMUNet, VMUNet_enhance


def factory(name: str, in_channels: int, out_classes: int):
    if name == "unet":
        return UNet(in_channels, out_classes, use_dropout=True)
    if name == "unet32":
        return UNet(in_channels, out_classes, num_filters=32, use_dropout=True)
    if name == "unet8":
        return UNet(in_channels, out_classes, num_filters=8, use_dropout=True)
    if name == "unet_nd":
        return UNet(in_channels, out_classes, use_dropout=False)
    if name == "unet8_nd":
        return UNet(in_channels, out_classes, num_filters=8, use_dropout=False)
    if name == "unet32_nd":
        return UNet(in_channels,
                    out_classes,
                    num_filters=32,
                    use_dropout=False)
    if name == "deeplab":
        return DeepLabV3Plus.from_pretrained(in_channels, out_classes)
    if name == "lightmunet":
        return LightMUNet(spatial_dims=2,
                          in_channels=in_channels,
                          out_channels=out_classes)
    if name == "lightmunet16":
        return LightMUNet(spatial_dims=2,
                          in_channels=in_channels,
                          out_channels=out_classes,
                          init_filters=16)
    if name == "VMUnet":
        return VMUNet(input_channels=in_channels,
                          num_classes=out_classes,                        
                          )
    if name == "VMUnet_enhance":
        return VMUNet_enhance(input_channels=in_channels,
                          num_classes=out_classes,                        
                          )
    raise NotImplementedError
