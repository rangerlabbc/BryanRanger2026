# =============================================================================
# model.py
# UNet for body composition regression from ultrasound images.
#
# Takes a list of image tensors (one per image) and outputs a single scalar
# prediction (FM or FFM).
#
# USAGE:
#   from maternal_ultrasound.model import UNet
#   model = UNet(n_channels=3, n_classes=1)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================

class DoubleConv(nn.Module):
    """Two consecutive (Conv → BatchNorm → LeakyReLU) blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """MaxPool then DoubleConv — one encoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upsample then DoubleConv — one decoder step.
    Concatenates the skip connection from the encoder.
    bilinear=True uses fixed upsampling (not learnable) which is
    more memory-efficient.
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if spatial sizes differ slightly due to odd input dimensions
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# =============================================================================
# UNet
# =============================================================================

class UNet(nn.Module):
    """
    UNet for regression: predicts a single scalar (FM or FFM) from
    a list of ultrasound image tensors.

    Architecture:
      Encoder: inc → down1 → down2 → down3 → down4
      Decoder: up1 → up2 → up3 → up4 → outconv
      Regression head: sigmoid → flatten → linear1 (256*256 → 1)
                       per-image outputs → linear2 (N_images → 1)

    Args:
        n_channels : number of input channels (3 for RGB)
        n_classes  : output channels of outconv (keep at 1)
        n_images   : expected number of images per patient.
                     Sets the size of the final linear layer.
                     Default 6 = 2 images × 3 regions (BQS).
                     Adjust if you change num_images or region_combo.
        bilinear   : use bilinear upsampling (True) or learnable
                     transposed conv (False)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1,
                 n_images: int = 6, bilinear: bool = True):
        super().__init__()

        self.n_images = n_images

        # Encoder
        self.inc    = DoubleConv(n_channels, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)
        self.down4  = Down(512, 512)

        # Decoder
        self.up1    = Up(1024, 256, bilinear)
        self.up2    = Up(512,  128, bilinear)
        self.up3    = Up(256,  64,  bilinear)
        self.up4    = Up(128,  64,  bilinear)

        # Output conv: maps to n_classes channels (1 channel, 256x256)
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Dropout after some encoder steps
        self.dropout = nn.Dropout2d(0.1)

        # Regression head
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 * 256, 1)   # per-image scalar
        self.linear2 = nn.Linear(n_images, 1)     # aggregate across images

        # Weight initialization
        self.apply(self._init_weights)

    def forward(self, images: list, weight=None, length=None) -> torch.Tensor:
        """
        Args:
            images : list of image tensors, each shape (1, C, H, W) or (C, H, W)
            weight : optional patient weight (unused by default)
            length : optional patient length (unused by default)

        Returns:
            Scalar prediction tensor
        """
        outputs = []
        for image in images:
            x = image.float()
            if x.dim() == 3:
                x = x.unsqueeze(0)  # add batch dim if missing

            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x2 = self.dropout(x2)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            # Decoder
            x  = self.up1(x5, x4)
            x  = self.up2(x,  x3)
            x  = self.dropout(x)
            x  = self.up3(x,  x2)
            x  = self.dropout(x)
            x  = self.up4(x,  x1)

            # Output + regression
            x = self.outconv(x)
            x = torch.sigmoid(x)
            x = self.flatten(x)
            x = self.linear1(x)
            outputs.append(x)

        # Stack all per-image predictions and aggregate to final scalar
        outputs = torch.cat(outputs, dim=0)   # shape: (n_images, 1)
        outputs = outputs.squeeze(1)           # shape: (n_images,)
        outputs = self.linear2(outputs)        # shape: (1,)
        return outputs

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.uniform_(module.weight, a=0.0, b=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
