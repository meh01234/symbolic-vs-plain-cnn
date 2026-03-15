# MNIST Addition: How Symbolic Constraints Shape CNN Representations

A controlled study comparing how a neuro-symbolic CNN and a plain CNN develop
internal representations when trained on MNIST addition using only sum labels.
No individual digit labels are provided to either model at any point.

## Research Question

When two models solve the same task with identical supervision, does imposing a
symbolic constraint on the architecture change what the CNN learns internally?

## Motivation

The human brain solves arithmetic through two functionally distinct systems.
The visual cortex processes raw digit images, recognising shapes and patterns.
A separate region, primarily the parietal cortex, handles the arithmetic itself,
working with abstract quantities rather than visual features.
These two systems are anatomically separate with a clean interface between them.

The NeSy model explicitly separates visual perception from symbolic reasoning, while the plain CNN does not. This project tests whether that architectural separation changes what visual features the CNN learns, measured through channel selectivity and channel-ablation sensitivity.



## Models

**PlainAdditionModel**
Images → DigitCNN → concatenate features → Linear layers → sum classes
No structural constraint. Visual processing and arithmetic reasoning are collapsed
into one undifferentiated network.

**NeSyAdditionModel**
Images → DigitCNN → Linear(256→10) → Softmax → digit probability distributions
→ Symbolic module computes P(sum=s) via explicit arithmetic rule table
Visual processing and arithmetic reasoning are architecturally separated.
The CNN is forced to produce digit probability distributions because the
symbolic module demands them as input.

**Shared DigitCNN backbone**
```
Conv2d(1→16, k=5, p=2) + BN + ReLU + MaxPool → (16, 14, 14)
Conv2d(16→32, k=3, p=1) + BN + ReLU + MaxPool → (32, 7, 7)
Flatten → Linear(1568→256) + ReLU → 256-dim embedding
```

## Experiment 1: Single-Digit Addition

Each example is two MNIST images. Label is their sum (0-18, 19 classes).

| Model | Epoch 0 Acc | Final Acc |
|-------|-------------|-----------|
| Plain CNN | 87.64% | 98.24% |
| NeSy CNN  | 96.44% | 98.52% |

Both models solve single-digit addition well. NeSy converges faster but both
reach similar final accuracy. The task is simple enough that the plain CNN
finds a good solution without symbolic structure, and the representations
reflect this: both models develop similar internal organisation.

| Metric | Plain CNN | NeSy CNN |
|--------|-----------|----------|
| Mean channel selectivity | 0.4483 | 0.4525 |
| Critical channels (>1% drop) | 0 | 0 |
| Max ablation drop | 0.50% | 0.75% |

When the task is easy, the symbolic constraint does not matter.

## Experiment 2: Two-Digit Addition

Each example is four MNIST images representing two 2-digit numbers A and B.
Label is A + B (0-198, 199 classes).

This task is substantially harder. The plain CNN must figure out place value
from raw pixels with no guidance. The NeSy CNN has place value built into
its symbolic module explicitly.

| Model | Epoch 0 Acc | Final Acc |
|-------|-------------|-----------|
| Plain CNN | 4.98%  | 93.70% |
| NeSy CNN  | 93.06% | 95.82% |

Plain CNN starts near chance and climbs slowly over 20 epochs.
NeSy CNN starts at 93% immediately because the symbolic module provides
correct arithmetic structure from the first forward pass.

| Metric | Plain CNN | NeSy CNN |
|--------|-----------|----------|
| Mean channel selectivity | 0.4401 | 0.4547 |
| Critical channels (>1% drop) | 10 | 5 |
| Max ablation drop | 2.50% | 2.10% |

When the task is hard, the symbolic constraint produces measurably different
internal organisation.


## Core Finding

**When the task is easy, the symbolic constraint does not matter.**
Single-digit addition is simple enough that the plain CNN finds a good solution
on its own. Both models end up with similar accuracy and similar internal representations.

**When the task is hard, the symbolic constraint changes what the CNN learns.**
Two-digit addition requires understanding place value. The plain CNN has no guidance
and spends 20 epochs searching for a solution, starting near chance.
The NeSy CNN has place value built into its symbolic module and solves the task
immediately from epoch 0.

This pressure difference leaves a measurable trace in the representations:

- Plain CNN channels respond broadly and inconsistently across digits.
  Many channels are critical but some actually hurt performance when removed,
  suggesting the model found a messy, fragile solution.

- NeSy CNN channels develop clear preferences for specific digits even though
  digit labels were never provided during training. Fewer channels are critical
  and all of them contribute positively, suggesting a cleaner, more organised solution.

**The symbolic constraint did not make individual channels more specialised on average.
It made the overall representation more coherent.** The difference is not in how
selective any single channel is, but in whether the channels work together in an
organised way or not.



## Limitations

Analysis is conducted on a small 32-channel CNN. Whether these representational
differences scale to larger architectures is an open question. Channel ablation
on 2000 test pairs introduces noise and some observed differences may not be
statistically significant without repeated trials across multiple seeds.
Single-digit results serve as a control condition rather than a finding in themselves.

## Training Details
```
Optimiser:     Adam lr=1e-3, weight_decay=1e-4
Scheduler:     ReduceLROnPlateau patience=3, factor=0.5
Gradient clip: 1.0
Epochs:        20
Batch size:    64
Single digit:  train pairs=60,000 / test=10,000
Two digit:     train pairs=30,000 / test=5,000
Seed:          0
Loss (Plain):  CrossEntropyLoss
Loss (NeSy):   NLLLoss on log(sum_probs)
```

## Reproducibility
```bash
pip install -r requirements.txt
python train.py
python analyse.py
```

All experiments use fixed seeds across Python, NumPy, and PyTorch with
deterministic CUDA operations enabled.

## File Structure
```
nesy_mnist/
├── cnn.py                    # Shared DigitCNN backbone
├── plain_model.py            # PlainAdditionModel
├── nesy_model.py             # NeSyAdditionModel with symbolic module
├── two_digit_dataset.py      # TwoDigitAdditionDataset
├── train.py                  # Training loop for both models
├── analyse.py                # Tuning curves and channel ablation
├── outputs/
│   ├── plain_model.pt
│   └── nesy_model.pt
└── figures/
    ├── tuning_curves.png
    ├── selectivity_distribution.png
    └── channel_ablation.png
```