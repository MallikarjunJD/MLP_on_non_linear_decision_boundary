# MLP on a Non-Linear Decision Boundary — PyTorch

**Step 2 of 4** in a series building toward Convolutional Neural Networks. Step 1
built an MLP from scratch to learn a linear function; this step keeps the same
mechanics and deliberately swaps in a target that is **not linearly separable**,
to test — and visually prove — why hidden layers and non-linear activations
matter at all.

## Why Synthetic Data (Again)

As in Step 1, this project uses a synthetic dataset generated from a known,
closed-form rule rather than a real-world dataset — here, an XOR-style pattern:
a point is labeled class 1 if its two coordinates have different signs, and
class 0 otherwise.

This is a deliberate engineering choice, not a limitation, and it does something
Step 1's linear dataset couldn't:

- **Ground truth is a known, non-linear shape.** The true decision boundary
  (two intersecting lines, forming four alternating quadrants) is exactly known
  in advance, so "did the model learn the right shape?" is a visual,
  checkable question rather than a guess.
- **It isolates a single architectural variable.** Because the data-generating
  process is fixed and controlled, any change in performance when removing the
  hidden layer's non-linearity (see the control experiment in the notebook) can
  be attributed directly to that architectural choice — not to noisy or
  ambiguous real-world labels.
- **It's the smallest known counterexample to linear models.** XOR is the
  textbook case used to demonstrate the limits of single-layer perceptrons; using
  it here makes the "why do we need non-linearity" argument concrete instead of
  theoretical.

In short: same philosophy as Step 1 — the dataset is a controlled instrument for
auditing the model, not the point of the exercise.

## Architecture

```
Input (2 features)
   │
   ▼
Linear(2 → 16)
   │
   ▼
ReLU
   │
   ▼
Linear(16 → 16)
   │
   ▼
ReLU
   │
   ▼
Linear(16 → 1)
   │
   ▼
Sigmoid
   │
   ▼
Output (probability, binary classification)
```

| Component | Choice | Rationale |
|---|---|---|
| Hidden layers | 2 | One is theoretically sufficient for XOR; two produces a cleaner, more visually interpretable decision boundary |
| Hidden units | 16 per layer | Enough capacity to resolve the boundary clearly without heavy overfitting risk |
| Activation | ReLU | Consistent with Step 1; deep-dive notes cover why it — specifically the non-linearity, not the layer count — is what makes XOR solvable |
| Output activation | Sigmoid | Converts a raw score into a class-1 probability |
| Loss | Binary Cross-Entropy | Standard loss for binary classification; pairs with Sigmoid for a clean combined gradient (see deep-dive notes) |
| Optimizer | Adam (lr=0.01) | Same choice as Step 1, for consistency across the series |

## Repository Structure

```
.
├── README.md
├── mlp_classification.ipynb   # main notebook (public-facing, documented)
├── src/
│   ├── data.py                 # synthetic XOR-pattern data generation
│   ├── model.py                 # MLP definition (2-hidden-layer classifier)
│   └── train.py                 # training loop + accuracy tracking
├── requirements.txt
└── LICENSE
```

## Engineering Standards

- **Reproducibility:** fixed random seeds for data generation and model
  initialization; dependencies pinned in `requirements.txt`.
- **Separation of concerns:** data generation, model definition, and training
  logic remain decoupled, consistent with Step 1's structure.
- **Inline documentation:** every layer, loss computation, and optimizer step is
  commented at the point of use.
- **Validation split + accuracy tracking:** loss alone doesn't fully capture
  classification quality near a decision boundary, so both are logged and
  plotted every run.
- **Built-in control experiment:** the notebook includes an ablation (removing
  all non-linear activations) as a first-class, reproducible step — not an
  afterthought — to make the architectural claim falsifiable rather than
  asserted.

## Quickstart

```bash
git clone https://github.com/MallikarjunJD/MLP_on_non_linear_decision_boundary.git
cd mlp-classification-boundary
pip install -r requirements.txt
jupyter notebook mlp_classification.ipynb
```

## Results

| Metric | Value |
|---|---|
| Validation accuracy (full model, with ReLU) | high — see notebook, Section 6 |
| Validation accuracy (linear-only control) | ~50% (chance level on a balanced set) — see notebook, Section 8 |
| Decision boundary shape | approximates the true X-shaped XOR boundary — see notebook, Section 7 |

Exact figures are logged in the notebook output and will vary slightly run to
run due to random initialization and mini-batch sampling.

## Series Roadmap

- [x] Step 1 — MLP from scratch: regression on a synthetic linear function
- [x] Step 2 — MLP on a classification task with a non-linear decision boundary
- [ ] Step 3 — CNN on image data



## Author

**Mallikarjun Jadi**

Computer Science Engineering Student

Machine Learning Engineer | Full Stack Developer

## License

MIT


