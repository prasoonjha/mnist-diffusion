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
cd /Users/prasoonjha/dev/ai/mnist-diffusion
python3 train.py
```

## Demo

![Generated 101](generated_101.png)
