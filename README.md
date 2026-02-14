# Conditional GAN for MNIST (PyTorch)

A minimal Conditional GAN (cGAN) project to generate MNIST digits with label control (`0-9`).

## Project Structure

- `model.py` — `Generator` and `Discriminator` with label conditioning via `nn.Embedding`
- `train.py` — MNIST download + cGAN training loop (20 epochs)
- `inference.py` — loads trained generator and generates digit-sequence images (e.g., `"101"`)

## Requirements

- Python 3.8+
- `pip`

Install dependencies:

```bash
python3 -m pip install torch torchvision
```

## Train

From project root:

```bash
python3 train.py
```

## Demo

<img src="generated_101.png" alt="Generated 101" width="700" />

## Architecture (cGAN)

This project uses a **Conditional GAN (cGAN)**, which means both the Generator and Discriminator are conditioned on a digit label (`0-9`).

### 1) Label conditioning

- Labels are represented using `nn.Embedding(num_classes=10, embedding_dim=...)`.
- The label embedding is learned jointly during training.
- Conditioning is applied to both networks:
  - **Generator:** receives noise vector `z` (100-dim) + label embedding
  - **Discriminator:** receives image `x` + label embedding

### 2) Generator

- Input:
  - `z ~ N(0, I)` of shape `[batch_size, 100]`
  - class label `y` of shape `[batch_size]`
- Process:
  - Embed `y` using `nn.Embedding`
  - Concatenate `[z, emb(y)]`
  - Pass through feed-forward layers to produce an MNIST-sized image
- Output:
  - tensor shaped like `1 x 28 x 28` (or flattened equivalent), then reshaped for image saving

Intuition: for the same label, different `z` values produce different writing styles of that digit.

### 3) Discriminator

- Input:
  - image `x` (real or generated)
  - class label `y`
- Process:
  - Flatten image (if needed)
  - Embed `y`
  - Concatenate `[x, emb(y)]`
  - Pass through feed-forward layers
- Output:
  - scalar probability/logit indicating whether `(x, y)` is real or fake

Intuition: the Discriminator must verify both image realism **and** label consistency.

### 4) Training objective

Training alternates between:

- **Discriminator step:**
  - classify real pairs `(x_real, y_real)` as real
  - classify fake pairs `(G(z, y), y)` as fake
- **Generator step:**
  - generate images `G(z, y)` that fool the Discriminator into predicting real

A standard BCE-based GAN objective is used in the training loop for 20 epochs.

### 5) Why this works for sequence generation

Because the Generator is label-conditional, inference can generate each character independently:

1. Read each digit in `number_string`
2. Sample a noise vector for each digit
3. Generate one `28x28` image per digit with its label
4. Concatenate horizontally to create one final sequence image

## Quick Start (No Training Required)

This repo ships a pretrained generator at:

- `release/generator.pth`

Run directly:

```bash
python3 inference.py 314159 generated_314159.png
```

Default behavior:
1. tries `checkpoints/generator.pth` (your locally trained weights)
2. falls back to `release/generator.pth` (pretrained shipped weights)

You can also pass a custom checkpoint:

```bash
python3 inference.py 314159 generated_314159.png --weights /path/to/generator.pth
```
