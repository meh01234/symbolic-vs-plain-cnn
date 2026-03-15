import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
FIGURE_DIR = Path('figures')
FIGURE_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load trained models
plain = PlainAdditionModel().to(device)
nesy  = NeSyAdditionModel().to(device)
plain.load_state_dict(torch.load(OUTPUT_DIR / 'plain_model.pt'))
nesy.load_state_dict(torch.load(OUTPUT_DIR  / 'nesy_model.pt'))
plain.eval()
nesy.eval()

# Load MNIST test images with digit labels for tuning curve analysis
transform   = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])
test_mnist  = datasets.MNIST('data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_mnist, batch_size=256, shuffle=False)

addition_loader = DataLoader(TwoDigitAdditionDataset(test_mnist, 2000, seed=1), batch_size=64)


def get_tuning_curves(model):
    # Hook post-ReLU activations, average over spatial dims, group by digit label.
    activations_by_digit = {d: [] for d in range(10)}

    def hook_fn(module, input, output):
        hook_fn.out = output.mean(dim=(2, 3)).detach().cpu()

    hook = model.cnn.block2[2].register_forward_hook(hook_fn)
    with torch.no_grad():
        for imgs, labels in test_loader:
            _ = model.cnn(imgs.to(device))
            for i, label in enumerate(labels):
                activations_by_digit[label.item()].append(hook_fn.out[i].numpy())
    hook.remove()

    # Mean activation per digit per channel → shape (10, 32)
    return np.stack([np.stack(activations_by_digit[d]).mean(axis=0)
                     for d in range(10)])


def selectivity(tuning):
    # Shift each channel so its weakest digit response is 0.
    # This avoids sign-related artifacts when activations are negative.
    shifted = tuning - tuning.min(axis=0, keepdims=True)
    max_r  = shifted.max(axis=0)
    mean_r = shifted.mean(axis=0)

    # Bounded in [0, 1]: 0 means flat responses, 1 means one-digit dominance.
    return (max_r - mean_r) / (max_r + 1e-10)


def evaluate(model, loader, is_nesy):
    correct, total = 0, 0
    with torch.no_grad():
        for a_tens, a_ones, b_tens, b_ones, label in loader:
            a_tens = a_tens.to(device)
            a_ones = a_ones.to(device)
            b_tens = b_tens.to(device)
            b_ones = b_ones.to(device)
            label = label.to(device)
            out = (
                model(a_tens, a_ones, b_tens, b_ones)[0]
                if is_nesy
                else model(a_tens, a_ones, b_tens, b_ones)
            )
            correct += (out.argmax(1) == label).sum().item()
            total   += label.size(0)
    return correct / total


def run_ablation(model, loader, baseline, is_nesy):
    # Zero out each post-pooling channel independently, measure accuracy drop.
    drops = []
    for ch in range(32):
        def hook_fn(module, input, output, ch=ch):
            output[:, ch, :, :] = 0
            return output
        hook = model.cnn.block2[3].register_forward_hook(hook_fn)
        acc  = evaluate(model, loader, is_nesy)
        drops.append(baseline - acc)
        hook.remove()
    return np.array(drops)


# Extract tuning curves
print("Extracting tuning curves...")
plain_tuning = get_tuning_curves(plain)
print("plain done")
nesy_tuning  = get_tuning_curves(nesy)
print("nesy done")
plain_sel = selectivity(plain_tuning)
nesy_sel  = selectivity(nesy_tuning)

print(f"Plain mean selectivity: {plain_sel.mean():.4f}")
print(f"NeSy  mean selectivity: {nesy_sel.mean():.4f}")

# Run ablation
print("\nRunning ablation...")
baseline_plain = evaluate(plain, addition_loader, is_nesy=False)
baseline_nesy  = evaluate(nesy,  addition_loader, is_nesy=True)
print(f"Baseline Plain={baseline_plain:.4f} | NeSy={baseline_nesy:.4f}")

plain_drops = run_ablation(plain, addition_loader, baseline_plain, is_nesy=False)
nesy_drops  = run_ablation(nesy,  addition_loader, baseline_nesy,  is_nesy=True)

print(f"Plain max drop: {plain_drops.max():.4f} | NeSy max drop: {nesy_drops.max():.4f}")

# Figure 1: Tuning curves for top 6 most selective NeSy channels
top6  = np.argsort(nesy_sel)[::-1][:6]
fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for col, ch in enumerate(top6):
    for row, (tuning, sel, label, color) in enumerate([
        (plain_tuning, plain_sel, 'Plain', 'steelblue'),
        (nesy_tuning,  nesy_sel,  'NeSy',  'darkorange')
    ]):
        axes[row, col].bar(range(10), tuning[:, ch], color=color)
        axes[row, col].set_title(f'Ch {ch} sel={sel[ch]:.2f}', fontsize=9)
        axes[row, col].set_xticks(range(10))
        if col == 0:
            axes[row, col].set_ylabel(f'{label}\nActivation')

plt.suptitle('Tuning Curves: Top 6 Most Selective NeSy Channels (Two-Digit Addition)\nPlain (top) vs NeSy (bottom)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'tuning_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved tuning_curves.png")

# Figure 2: Selectivity distribution
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 20)
ax.hist(plain_sel, bins=bins, alpha=0.6, color='steelblue',  label=f'Plain mean={plain_sel.mean():.3f}')
ax.hist(nesy_sel,  bins=bins, alpha=0.6, color='darkorange', label=f'NeSy  mean={nesy_sel.mean():.3f}')
ax.axvline(plain_sel.mean(), color='steelblue',  linestyle='--')
ax.axvline(nesy_sel.mean(),  color='darkorange', linestyle='--')
ax.set_xlabel('Selectivity Score')
ax.set_ylabel('Number of Channels')
ax.set_title('Channel Selectivity Distribution (Two-Digit Addition): Plain CNN vs NeSy CNN')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'selectivity_distribution.png', dpi=150, bbox_inches='tight')
print("Saved selectivity_distribution.png")

# Figure 3: Channel ablation comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

for ax, drops, label, color in [
    (ax1, plain_drops, 'Plain CNN', 'steelblue'),
    (ax2, nesy_drops,  'NeSy CNN',  'darkorange')
]:
    colours = ['#d62728' if v > 0.01 else color for v in drops]
    ax.bar(range(32), drops, color=colours)
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='1% threshold')
    ax.axhline(y=0,    color='black', linewidth=0.5)
    ax.set_title(f'{label}: Accuracy Drop per Channel')
    ax.set_xlabel('Conv2 Channel Index')
    ax.set_ylabel('Accuracy Drop')
    ax.legend()

plt.suptitle('Channel Ablation (Two-Digit Addition): Plain CNN vs NeSy CNN\nRed = causally important (>1% drop)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'channel_ablation.png', dpi=150, bbox_inches='tight')
print("Saved channel_ablation.png")

# Summary
print("\n=== Summary ===")
print(f"Plain mean selectivity:          {plain_sel.mean():.4f}")
print(f"NeSy  mean selectivity:          {nesy_sel.mean():.4f}")
plain_mean_sel = plain_sel.mean()
nesy_mean_sel = nesy_sel.mean()
if abs(plain_mean_sel) > 1e-12:
    sel_increase_pct = (nesy_mean_sel - plain_mean_sel) / plain_mean_sel * 100
    print(f"Selectivity increase:            {sel_increase_pct:.1f}%")
else:
    print("Selectivity increase:            n/a (plain mean near zero)")
print(f"Plain critical channels (>1%):   {(plain_drops > 0.01).sum()}")
print(f"NeSy  critical channels (>1%):   {(nesy_drops  > 0.01).sum()}")
print(f"Plain max ablation drop:         {plain_drops.max():.4f}")
print(f"NeSy  max ablation drop:         {nesy_drops.max():.4f}")



