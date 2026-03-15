import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import DigitCNN


class NeSyAdditionModel(nn.Module):
    """
    Neuro-symbolic CNN for two-digit MNIST addition.
    Takes four MNIST images corresponding to A=(a_tens, a_ones)
    and B=(b_tens, b_ones), then routes through explicit carry-aware
    symbolic addition.
    CNN is forced to output digit probability distributions
    because the symbolic module requires them to predict the sum.
    Never told individual digit labels, only the sum.
    """
    def __init__(self):
        super().__init__()
        self.cnn = DigitCNN()

        # Maps 256-dim features to digit probabilities (0-9)
        self.digit_classifier = nn.Linear(256, 10)

        # Fixed symbolic rule table for two-digit addition with carry.
        # sum_table[s, at, ao, bt, bo] = 1 iff
        # s == (10*at + ao) + (10*bt + bo)
        sum_table = torch.zeros(199, 10, 10, 10, 10, dtype=torch.float32)
        for at in range(10):
            for ao in range(10):
                for bt in range(10):
                    for bo in range(10):
                        total = (10 * at + ao) + (10 * bt + bo)
                        sum_table[total, at, ao, bt, bo] = 1.0
        self.register_buffer("sum_table", sum_table)

    def forward(self, a_tens, a_ones, b_tens, b_ones):
        # Extract features from each digit slot.
        f_at = self.cnn(a_tens)  # (batch, 256)
        f_ao = self.cnn(a_ones)  # (batch, 256)
        f_bt = self.cnn(b_tens)  # (batch, 256)
        f_bo = self.cnn(b_ones)  # (batch, 256)

        # Convert features to digit probability distributions.
        p_at = F.softmax(self.digit_classifier(f_at), dim=1)  # (batch, 10)
        p_ao = F.softmax(self.digit_classifier(f_ao), dim=1)  # (batch, 10)
        p_bt = F.softmax(self.digit_classifier(f_bt), dim=1)  # (batch, 10)
        p_bo = F.softmax(self.digit_classifier(f_bo), dim=1)  # (batch, 10)

        # Independence across slots gives a factorized joint over 4 digits.
        # Then symbolic table maps joint digit assignments to final sum class.
        sum_probs = torch.einsum(
            "na,nb,nc,nd,sabcd->ns",
            p_at,
            p_ao,
            p_bt,
            p_bo,
            self.sum_table,
        )  # (batch, 199)

        return sum_probs, p_at, p_ao, p_bt, p_bo