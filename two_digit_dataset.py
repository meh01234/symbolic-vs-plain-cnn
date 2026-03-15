import numpy as np
from torch.utils.data import Dataset


class TwoDigitAdditionDataset(Dataset):
    """
    Generates pairs of two-digit numbers from MNIST digit images.

    Returns four images per sample:
    - a_tens, a_ones for first addend A
    - b_tens, b_ones for second addend B
    and label A + B in [0, 198].
    """

    def __init__(self, mnist_dataset, n_pairs, seed=0):
        self.mnist = mnist_dataset
        self.n_pairs = n_pairs

        rng = np.random.RandomState(seed)
        n = len(mnist_dataset)

        self.a_tens_idx = rng.randint(0, n, n_pairs)
        self.a_ones_idx = rng.randint(0, n, n_pairs)
        self.b_tens_idx = rng.randint(0, n, n_pairs)
        self.b_ones_idx = rng.randint(0, n, n_pairs)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        a_tens_img, a_tens_digit = self.mnist[self.a_tens_idx[idx]]
        a_ones_img, a_ones_digit = self.mnist[self.a_ones_idx[idx]]
        b_tens_img, b_tens_digit = self.mnist[self.b_tens_idx[idx]]
        b_ones_img, b_ones_digit = self.mnist[self.b_ones_idx[idx]]

        a_value = 10 * a_tens_digit + a_ones_digit
        b_value = 10 * b_tens_digit + b_ones_digit
        sum_label = a_value + b_value  # 0..198

        return a_tens_img, a_ones_img, b_tens_img, b_ones_img, sum_label
