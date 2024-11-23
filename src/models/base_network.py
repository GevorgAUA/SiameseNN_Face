import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        """
        Convolutional base network for feature extraction.
        """
        super(BaseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Embedding size
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
