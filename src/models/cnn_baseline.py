import torch
import torch.nn as nn


class SpeechEnhancementCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ============================================
        # Encoder / Feature Extraction
        # ============================================

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ============================================
        # Feature Reconstruction
        # ============================================

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

        # ============================================
        # Activation
        # ============================================

        self.relu = nn.ReLU()

    def forward(self, x):

        # --------------------------------------------
        # Layer 1
        # --------------------------------------------

        x = self.conv1(x)

        x = self.relu(x)

        # --------------------------------------------
        # Layer 2
        # --------------------------------------------

        x = self.conv2(x)

        x = self.relu(x)

        # --------------------------------------------
        # Layer 3
        # --------------------------------------------

        x = self.conv3(x)

        x = self.relu(x)

        # --------------------------------------------
        # Output layer
        # --------------------------------------------

        x = self.conv4(x)

        return x