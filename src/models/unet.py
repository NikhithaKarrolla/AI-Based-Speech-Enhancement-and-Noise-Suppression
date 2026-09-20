import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with
    BatchNorm and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SpeechEnhancementUNet(nn.Module):
    """
    U-Net architecture for speech enhancement.

    Input:
        [batch, 1, frequency, time]

    Output:
        [batch, 1, frequency, time]
    """

    def __init__(self):
        super().__init__()

        # =========================
        # Encoder
        # =========================

        self.enc1 = DoubleConv(1, 16)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )

        self.enc2 = DoubleConv(16, 32)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )

        self.enc3 = DoubleConv(32, 64)

        self.pool3 = nn.MaxPool2d(
            kernel_size=2
        )

        # =========================
        # Bottleneck
        # =========================

        self.bottleneck = DoubleConv(
            64,
            128
        )

        # =========================
        # Decoder
        # =========================

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            128,
            64
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            64,
            32
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            32,
            16
        )

        # =========================
        # Output
        # =========================

        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def _resize_to_match(self, x, reference):
        """
        Resize x to the spatial dimensions of reference.

        This handles odd dimensions such as the
        257 frequency bins produced by STFT.
        """

        if x.shape[-2:] != reference.shape[-2:]:

            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return x

    def forward(self, x):

        # =========================
        # Encoder
        # =========================

        e1 = self.enc1(x)

        p1 = self.pool1(e1)

        e2 = self.enc2(p1)

        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        p3 = self.pool3(e3)

        # =========================
        # Bottleneck
        # =========================

        b = self.bottleneck(p3)

        # =========================
        # Decoder 3
        # =========================

        d3 = self.up3(b)

        d3 = self._resize_to_match(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        # =========================
        # Decoder 2
        # =========================

        d2 = self.up2(d3)

        d2 = self._resize_to_match(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        # =========================
        # Decoder 1
        # =========================

        d1 = self.up1(d2)

        d1 = self._resize_to_match(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # =========================
        # Output
        # =========================

        output = self.output(d1)

        # Guarantee exact input spatial dimensions
        output = self._resize_to_match(
            output,
            x
        )

        return output