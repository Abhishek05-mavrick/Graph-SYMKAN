# Graph-SYMBA

**Physics-Informed Graph Neural Network for Squared Amplitude Calculation**

A hybrid GNN + Transformer architecture that computes symbolic squared amplitudes from Feynman diagrams for the [ML4SCI](https://ml4sci.org/) SYMBA project. Developed as part of Google Summer of Code 2026.

---

## Overview

Computing squared amplitudes |M|² is a foundational step in linking theoretical particle physics predictions to experimental measurements. Traditional tools like FeynCalc are precise but require extensive manual symbolic setup. Previous ML approaches (SYMBA, SineKAN-Transformer) automate this via sequence-to-sequence models — but serialising Feynman diagrams into 1D token strings forces the model to relearn graph symmetry from data alone.

**Graph-SYMBA eliminates this mismatch.** It parses Feynman diagrams directly into PyTorch Geometric graph objects and encodes them with a physics-informed GNN before passing to an autoregressive Transformer decoder. Permutation equivariance is enforced as a hard structural prior — not a soft learned one.

### Key results (QED 2→2, 910 records, 80/10/10 split)

| Architecture Configuration | Edge Activation | Greedy Decoding (Seq. Accuracy) | Beam Search (k=5) (Seq. Accuracy) |
|---|---|---|---|
| Baseline Graph-Transformer | ReLU (Piecewise Linear) | 35.48% | 38.71% |
| Graph-SYMBA (SineKAN) | Learnable sinosuidal basis function | 34.41% | 35.48% |
| Graph-SYMBA (SIREN) | Periodic Sine sin(omega*x) | **40.86%** | **41.94%** |

Sequence accuracy equals symbolic accuracy across all runs — a consequence of the tight 40-token normalisation vocabulary.

---

## Architecture

```
Feynman Topology String
        │
        ▼
┌─────────────────────┐
│  Algebraic Sanitiser│  → 40-token normalised vocabulary
│  (physics-aware)    │    INDEX_*, MOMENTUM_*, <GAMMA>, <S>, <T>, <U>
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  topology_to_pyg    │  → PyTorch Geometric Data object
│  parser             │    nodes: particles (spin, mass, type, kinematic state)
│                     │    edges: fully-connected clique per vertex (8 physics features)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  GNN Encoder        │  → 3-layer message passing
│                     │    + absolute node index embedding
│                     │    + random walk positional encoding (RWSE, 16 steps)
│                     │    edge activation: SIREN / SineKAN / ReLU (ablation)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Topological Memory │  → to_dense_batch: variable graph → fixed [B, N_max, 256]
│  Bridge             │    binary padding mask passed to cross-attention
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Autoregressive     │  → 4-layer Pre-LayerNorm Transformer decoder
│  Transformer        │    cross-attends to immutable graph memory at every step
│  Decoder            │    greedy decoding + beam search (k=5)
└─────────────────────┘
        │
        ▼
   Symbolic Squared Amplitude
        │
        ▼
┌─────────────────────┐
│  SymPy Validation   │  → mathematical equivalence check (x+y == y+x)
│  Layer              │
└─────────────────────┘
```

### Node features (64-dim)

| Dimension | Feature | Example values |
|---|---|---|
| 0 | Particle type | 1.0 = electron, 2.0 = photon |
| 1 | Kinematic state | 1.0 = off-shell, 0.0 = on-shell |
| 2 | Quantum spin | 0.5 = fermion, 1.0 = boson |
| 3 | Reserved | — |
| 4–63 | Zero-padded | Available for future features |

### Edge features (32-dim, first 8 active)

| Index | Feature |
|---|---|
| 0 | Source particle type |
| 1 | Target particle type |
| 2 | Interaction type (fermion-boson, same-same) |
| 3 | Momentum product |
| 4 | Spin product (spin_src × spin_tgt) |
| 5 | Self-loop indicator |
| 6 | Same-type flag |
| 7 | Same momentum state flag |
| 8–31 | Zero-padded |

---

## Installation

```bash
git clone https://github.com/Abhishek05-mavrick/Graph-SYMBA.git
cd Graph-SYMBA
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- SymPy
- Weights & Biases (optional, for logging)

---

## Usage

### 1. Preprocess the dataset

```bash
python SYMBA-GSoC2026/src/data/preprocess_qed.py
```
This runs the algebraic sanitiser, normalising dummy indices, momentum labels, and physics tokens to the 40-token vocabulary.

### 2. Build the PyG graph dataset

```bash
python SYMBA-GSoC2026/src/data/topology_parser.py
```

### 3. Train

```bash
# SIREN (recommended)
python SYMBA-GSoC2026/src/train/train_graph_siren.py

# SineKAN
python SYMBA-GSoC2026/src/train/train_graph_sinekan.py

# Baseline Graph-Transformer (MLP)
python SYMBA-GSoC2026/src/train/train_graph_mlp.py
```

Training logs loss, token accuracy, sequence accuracy, and symbolic accuracy every 5 epochs. Checkpoints saved to `SYMBA-GSoC2026/checkpoints/`.

### 4. Evaluate and Inference

To evaluate the models and inspect predictions, explore the notebooks:

- `SYMBA-GSoC2026/Notebooks/Eval_and_inference.ipynb`

Evaluations report greedy and beam search sequence accuracy, plus SymPy symbolic equivalence on the test set.

---

## Project Structure

```
.
├── SYMBA-GSoC2026/
│   ├── Notebooks/
│   │   ├── Eval_and_inference.ipynb   # Greedy + beam decoding, accuracy metrics
│   │   ├── Preprocess.ipynb           # Data pipeline walkthrough
│   │   └── Training.ipynb             # Architecture comparison & run notebook
│   ├── src/
│   │   ├── data/
│   │   │   ├── preprocess_qed.py      # Algebraic sanitiser (40-token vocab)
│   │   │   └── topology_parser.py     # Feynman topology → PyG graph
│   │   ├── models/
│   │   │   ├── encoder.py             # GNN with SIREN/SineKAN/ReLU message passing
│   │   │   ├── decoder.py             # Pre-LayerNorm Transformer decoder
│   │   │   ├── components.py          # Node/Edge features and topological memory
│   │   │   └── graphSYMBA.py          # Full Model
│   │   └── train/
│   │       ├── metrics.py             # Accuracy and sympy metrics
│   │       ├── train_graph_siren.py   # Training script for SIREN
│   │       ├── train_graph_sinekan.py # Training script for SineKAN
│   │       ├── train_graph_mlp.py     # Training script for standard transformer
│   │       ├── train_seq2seq.py       # Baseline translation model
│   │       └── training.py            # Main training loop abstractions
│   ├── checkpoints/                   # Saved model weights
│   ├── preprocessed/                  # Processed graph files
│   ├── QCD data/                      # Raw QCD Dataset
│   └── QED data/                      # Raw QED Dataset
├── requirements.txt
└── README.md
```

---

## Data Normalisation

The algebraic sanitiser converts raw SYMBA records into a compact 40-token vocabulary before graph construction.

| Raw form | Normalised token | Physics meaning |
|---|---|---|
| `%sigma_165`, `%gam_145` | `INDEX_0`, `INDEX_1` | Dummy state indices (positional) |
| `j_1`, `k_2`, `l_3` | `MOMENTUM_0`, `MOMENTUM_1` | Momentum labels (positional) |
| `gamma_{+INDEX_0, ...}` | `<GAMMA>` | Dirac gamma matrix trace |
| `s_{12}`, `s_{13}` | `<S>` | Mandelstam s-channel variable |
| `t_{13}`, `t_{24}` | `<T>` | Mandelstam t-channel variable |
| `u_{14}` | `<U>` | Mandelstam u-channel variable |
| `m_e^2`, `m_mu^2` | `m_e`, `m_mu` | Physical mass (preserved exactly) |
| `e^4`, `e^2` | `e^4`, `e^2` | Coupling constant (preserved exactly) |

---

## Comparison Study

Three message-passing variants were evaluated on the QED 2→2 dataset (910 records, 80/10/10 split). All other hyperparameters held constant; learning rate reduced to 3×10⁻⁴ for sinusoidal variants to prevent gradient explosion from sin(ω·Wx) saturation.

**SIREN** (Graph-SYMBA recommended): A 3-layer sinusoidal MLP processes `[x_i | x_j | edge_attr]` in the message function. Fixed-frequency activations (ω=30) encode the oscillatory structure of QFT propagators directly. Achieves the highest final accuracy (43.01% beam search) and stable convergence.

**SineKAN**: A Kolmogorov-Arnold Network layer places learnable sinusoidal bases on the edge. Most expressive option but requires more epochs to stabilise under limited data. Tied the baseline on beam search at 80 epochs (40.86%) — expected to benefit most from larger datasets.

**Graph-Transformer baseline**: PyG's TransformerConv with ReLU activations, edge_dim=32, 4 heads. Fastest convergence, competitive beam search accuracy (40.86%). Recommended fallback.

SineKAN led SIREN in validation accuracy through epoch 35 (46.2% vs 35.2%), but SIREN overtook it by epoch 50. This confirms that SIREN converges to a better optimum on this task, while SineKAN's learnable bases require more epochs to stabilise — consistent with prior SineKAN literature.

---

## Acknowledgements

This project builds on the SYMBA codebase and is developed under the ML4SCI organisation as part of Google Summer of Code 2026. Architecture direction validated by mentor Eric Reinhardt (University of Alabama), who confirmed alignment with the team's internal research.

**References:**

- Buhler et al. SYMBA: Symbolic Computation of Squared Amplitudes. arXiv:2206.08901 (2022)
- Mitchell et al. Learning Feynman Diagrams using GNNs. arXiv:2211.15348 (2022)
- Reinhardt et al. SineKAN: KAN using Sinusoidal Activation Functions. Frontiers in AI 7 (2025)
- Sitzmann et al. Implicit Neural Representations with Periodic Activation Functions (SIREN). NeurIPS (2020)
- Fey & Lenssen. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop (2019)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
