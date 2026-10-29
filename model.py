import torch.nn as nn


class DigitsCNN(nn.Module):
    """Small CNN for 28x28 grayscale digits.

    1x28x28 -> Conv(32) -> Pool -> 32x14x14 -> Conv(64) -> Pool -> 64x7x7
            -> Flatten (3136) -> Linear(128) -> Linear(10)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
