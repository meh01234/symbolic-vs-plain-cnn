import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from plain_model import PlainAdditionModel
from nesy_model import NeSyAdditionModel
from two_digit_dataset import TwoDigitAdditionDataset

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_mnist = datasets.MNIST('data', train=True,  download=True, transform=transform)
test_mnist  = datasets.MNIST('data', train=False, download=True, transform=transform)

print(f"MNIST train: {len(train_mnist)} | test: {len(test_mnist)}")


# Create datasets
train_dataset = TwoDigitAdditionDataset(train_mnist, n_pairs=30000, seed=0)
test_dataset  = TwoDigitAdditionDataset(test_mnist,  n_pairs=5000,  seed=1)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def train_model(model, model_name, epochs=20):
    model = model.to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\n=== Training {model_name} ===")

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0

        for a_tens, a_ones, b_tens, b_ones, sum_label in train_loader:
            a_tens = a_tens.to(device)
            a_ones = a_ones.to(device)
            b_tens = b_tens.to(device)
            b_ones = b_ones.to(device)
            sum_label = sum_label.to(device)

            optimizer.zero_grad()

            if model_name == 'NeSy':
                sum_probs, _, _, _, _ = model(a_tens, a_ones, b_tens, b_ones)
                # NeSy returns probabilities, so train with NLL on log-probabilities.
                loss = nn.NLLLoss()(torch.log(sum_probs.clamp_min(1e-12)), sum_label)
            else:
                logits = model(a_tens, a_ones, b_tens, b_ones)
                loss = ce_criterion(logits, sum_label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for a_tens, a_ones, b_tens, b_ones, sum_label in test_loader:
                a_tens = a_tens.to(device)
                a_ones = a_ones.to(device)
                b_tens = b_tens.to(device)
                b_ones = b_ones.to(device)
                sum_label = sum_label.to(device)

                if model_name == 'NeSy':
                    sum_probs, _, _, _, _ = model(a_tens, a_ones, b_tens, b_ones)
                    preds = sum_probs.argmax(dim=1)
                else:
                    logits = model(a_tens, a_ones, b_tens, b_ones)
                    preds  = logits.argmax(dim=1)

                correct += (preds == sum_label).sum().item()
                total   += sum_label.size(0)

        acc = correct / total
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} | Acc={acc:.4f}")

    # Final evaluation
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for a_tens, a_ones, b_tens, b_ones, sum_label in test_loader:
            a_tens = a_tens.to(device)
            a_ones = a_ones.to(device)
            b_tens = b_tens.to(device)
            b_ones = b_ones.to(device)
            sum_label = sum_label.to(device)

            if model_name == 'NeSy':
                sum_probs, _, _, _, _ = model(a_tens, a_ones, b_tens, b_ones)
                preds = sum_probs.argmax(dim=1)
            else:
                logits = model(a_tens, a_ones, b_tens, b_ones)
                preds  = logits.argmax(dim=1)

            correct += (preds == sum_label).sum().item()
            total   += sum_label.size(0)

    final_acc = correct / total
    print(f"\n{model_name} Final Accuracy: {final_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), OUTPUT_DIR / f'{model_name.lower()}_model.pt')
    print(f"Saved {model_name.lower()}_model.pt")

    return model, final_acc


# Train both models
plain_model = train_model(PlainAdditionModel(), 'Plain', epochs=20)
nesy_model  = train_model(NeSyAdditionModel(),  'NeSy',  epochs=20)

print("\n=== Summary ===")
print(f"Plain final accuracy: {plain_model[1]:.4f}")
print(f"NeSy  final accuracy: {nesy_model[1]:.4f}")