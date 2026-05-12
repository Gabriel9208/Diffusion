# Conditional Diffusion Model on ICLEVR

A conditional image generation pipeline built on **DDPM** (training) + **DDIM** (inference) for the [ICLEVR](https://github.com/google/clevr-dataset-gen) dataset. The model generates 64×64 images conditioned on multi-label object descriptions (e.g. _"red cube, green sphere"_) using Classifier-Free Guidance and FiLM-based label injection.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [U-Net Backbone](#u-net-backbone)
  - [Time Embedding](#time-embedding)
  - [Label Embedding (FiLM)](#label-embedding-film)
  - [Noise Scheduling](#noise-scheduling)
  - [Classifier-Free Guidance](#classifier-free-guidance)
- [Training](#training)
- [Inference (DDIM)](#inference-ddim)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Experiments & Ablations](#experiments--ablations)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project implements a **text-conditioned diffusion model** that synthesises images of 3D objects. Key design choices:

| Component            | Choice                                                       |
| -------------------- | ------------------------------------------------------------ |
| Forward process      | DDPM with **linear** noise schedule (T = 1000)               |
| Backward / inference | **DDIM** with 50 steps (deterministic, σ = 0)                |
| Backbone             | **U-Net** with residual blocks                               |
| Condition encoding   | **Multi-hot vector** → single Linear layer                   |
| Condition injection  | **FiLM** (Feature-wise Linear Modulation)                    |
| Guidance             | **Classifier-Free Guidance** (cfg_scale = 3, p_uncond = 10%) |
| Evaluation           | FID via `clean-fid`                                          |

---

## Architecture

### U-Net Backbone

The backbone is a **U-Net** composed of residual blocks. Each residual block contains:

```
Conv → GroupNorm → SiLU  (×2)
```

with a shortcut `1×1 Conv` when input/output channels differ, otherwise an Identity.

### Time Embedding

To avoid spectral bias, the scalar timestep `t` is first encoded with a **sinusoidal embedding** (similar to NeRF positional encoding / transformer PE), then projected through MLP layers. The resulting embedding is injected _between_ GroupNorm and SiLU in each residual block layer.

```
t (scalar) → Sinusoidal → MLP → Linear (per-block) → inserted after GN
```

### Label Embedding (FiLM)

Object labels are encoded as a **sparse multi-hot vector** and projected through a single linear layer. Condition injection uses **FiLM**:

- An extra MLP doubles the channel size of the label embedding.
- The first half serves as **γ** (scale) and the second half as **β** (shift).
- Applied element-wise to the output of the first `(Conv → GN → SiLU)` sub-layer in every residual block.

```
labels → multi-hot → Linear → Linear (per-block) → FiLM (γ, β)
```

### Noise Scheduling

Linear schedule over **T = 1000** steps:

```
β_start = 1e-4,  β_end = 0.02
```

May try cosine scheduling (currently implemented but not used).

### Classifier-Free Guidance

During training, **10%** of samples drop the label condition (unconditional training). At inference, the model output is blended:

```
ε_guided = ε_uncond + cfg_scale × (ε_cond - ε_uncond)
```

`cfg_scale = 3` worked best; lower values produced noisy outputs, higher values caused over-saturation or noisy outputs.

---

## Training

The training objective follows the **simplified DDPM loss** — predicting the noise `ε` added at step `t`:

```
L = ||ε - ε_θ(x_t, t, c)||²
```

| Hyperparameter   | Value                           |
| ---------------- | ------------------------------- |
| Epochs           | 1000                            |
| Batch size       | 32                              |
| Learning rate    | 1e-4 (AdamW)                    |
| LR schedule      | Cosine Annealing (η_min = 1e-5) |
| Image resolution | 64 × 64                         |

Training is logged with **Weights & Biases**. Checkpoints are saved as `latest.pt` and `best.pt`.

---

## Inference (DDIM)

DDIM breaks the Markov chain assumption of DDPM, enabling **deterministic** sampling by skipping timesteps. With σ = 0, the update rule is:

```
x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1−ᾱ_{t-1}) · ε_θ
```

Only **50 steps** are used at inference (instead of 1000), drastically reducing sampling time while maintaining quality.

---

## Data Preprocessing

Images are preprocessed with a **LetterBox** transform:

1. Keep the original aspect ratio.
2. Scale the longer edge to **64 px**.
3. Pad the short edge with **grey** to produce a 64×64 image.

Final normalisation: `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)`.

---

## Experiments & Ablations

| Experiment          | Finding                                                              |
| ------------------- | -------------------------------------------------------------------- |
| **CFG scale**       | scale 3–4 is optimal; too low → noisy, too high → saturated          |
| **Label encoding**  | Multi-hot > CLIP embeddings (discrete separation is easier to learn) |
| **Label injection** | FiLM > additive (avoids time embedding dominating the signal)        |
| **Model size**      | Residual U-Net without attention; attention left for future work     |

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd Diffusion

# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python run.py \
  --mode train \
  --epochs 1000 \
  --batch-size 32 \
  --lr 1e-4 \
  --cfg-scale 3 \
  --cfg-p-uncond 0.1 \
  --latest-path ./results/latest.pt \
  --best-path ./results/best.pt \
  --save-img-dir ./images/run/ \
  --validate-every-epoch 10
```

### Resume Training

```bash
python run.py --mode train --resume --resume-path ./results/latest.pt
```

### Generate Result Grid

```bash
python run.py --mode grid --test-path test.json
# or
python run.py --mode grid --test-path new_test.json
```

### Visualise Denoising Process

```bash
python run.py --mode denoising
```

### Evaluate (Sweep Checkpoints)

```bash
python run.py --mode sweep
```

---

## Project Structure

```
Diffusion/
├── run.py                    # Entry point
├── src/
│   ├── dataset/
│   │   └── ICLVERDataset.py
│   ├── engine/
│   │   ├── Runner.py         # Training / inference loop
│   │   └── Callback.py       # Checkpoint & logging callbacks
│   ├── model/
│   │   ├── DDPM.py           # DDPM/DDIM model
│   │   ├── UNet.py           # U-Net backbone
│   │   └── condition_encoder/
│   │       └── MultiHotEncoder.py
│   └── utils/
│       ├── LetterBox.py
│       └── tarin_valid_split.py
├── train.json                # Training annotations
├── test.json                 # Test prompts
├── new_test.json             # Additional test prompts
├── objects.json              # Object class list
├── requirements.txt
└── README.md
```

---

## References

- [DDPM — Ho et al., 2020](https://arxiv.org/abs/2006.11239)
- [DDIM — Song et al., 2020](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Diffusion Guidance — Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)
- [FiLM — Perez et al., 2018](https://arxiv.org/abs/1709.07871)
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
