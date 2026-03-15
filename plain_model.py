import torch
import torch.nn as nn
from cnn import DigitCNN


class PlainAdditionModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cnn = DigitCNN()

        # Takes features from four digit slots concatenated
        # 256 * 4 = 1024 dimensional input
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 199)  # sums 0-198
        )

    def forward(self, a_tens, a_ones, b_tens, b_ones):
        # Extract features from each digit-slot image independently.
        f_at = self.cnn(a_tens)  # (batch, 256)
        f_ao = self.cnn(a_ones)  # (batch, 256)
        f_bt = self.cnn(b_tens)  # (batch, 256)
        f_bo = self.cnn(b_ones)  # (batch, 256)

        # Concatenate and predict two-digit sum directly.
        combined = torch.cat([f_at, f_ao, f_bt, f_bo], dim=1)  # (batch, 1024)
        return self.classifier(combined)  # (batch, 199)