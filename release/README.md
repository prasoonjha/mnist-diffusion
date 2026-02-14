# Release Weights

This folder contains pretrained weights for inference without training.

- `generator.pth` — pretrained MNIST cGAN generator checkpoint

Usage from project root:

```bash
python3 inference.py 314159 generated_314159.png
```

Optional explicit path:

```bash
python3 inference.py 314159 generated_314159.png --weights release/generator.pth
```