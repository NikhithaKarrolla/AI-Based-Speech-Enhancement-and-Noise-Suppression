import torch
import torch.nn as nn


class SpectrogramMSELoss(nn.Module):
    """
    Mean Squared Error loss for speech-enhancement
    spectrogram prediction.
    """

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, predicted, target):

        loss = self.mse(predicted, target)

        return loss