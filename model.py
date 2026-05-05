import torch.nn as nn


class PosterCNN(nn.Module):
    """Simple CNN for multi-label movie genre classification from posters."""

    def __init__(self, num_labels=20):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # (B, 32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (B, 32, 112, 112)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (B, 64, 56, 56)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (B, 128, 28, 28)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),      # (B, 128, 1, 1)
            nn.Flatten(),                      # (B, 128)
            nn.Linear(128, num_labels)         # (B, 20)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x

