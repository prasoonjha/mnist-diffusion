import argparse
import os
from typing import Optional, Tuple
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import Generator


def load_generator(
    checkpoint_path: str = "checkpoints/generator.pth",
    latent_dim: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple[Generator, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=latent_dim).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator, device


@torch.no_grad()
def generate_sequence(
    number_string: str,
    generator: Generator,
    device: torch.device,
    latent_dim: int = 100,
) -> torch.Tensor:
    if len(number_string) == 0:
        raise ValueError("number_string must not be empty")
    if not number_string.isdigit():
        raise ValueError("number_string must contain only digits 0-9")

    digit_images = []
    for digit_char in number_string:
        label = torch.tensor([int(digit_char)], dtype=torch.long, device=device)
        noise = torch.randn(1, latent_dim, device=device)
        generated = generator(noise, label)
        digit_images.append(generated[0].cpu())

    tiled = torch.cat(digit_images, dim=2)
    tiled = (tiled + 1.0) / 2.0
    return tiled.clamp(0.0, 1.0)


DEFAULT_WEIGHTS = "checkpoints/generator.pth"
RELEASE_WEIGHTS = "release/generator.pth"

def resolve_weights_path(user_path: str | None = None) -> str:
    if user_path:
        return user_path
    # Keep trained/checkpoints as default behavior, fallback to release for fresh users
    if Path(DEFAULT_WEIGHTS).exists():
        return DEFAULT_WEIGHTS
    return RELEASE_WEIGHTS


def main(number_string: str = "101", out_path: str = "generated_sequence.png", weights_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_weights_path(weights_path)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(ckpt, map_location=device))
    generator.eval()

    img = generate_sequence(number_string, generator, device)
    img.save(out_path)
    print(f"Saved: {out_path} (weights: {ckpt})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", nargs="?", default="101", help="Digit sequence, e.g. 314159")
    parser.add_argument("out", nargs="?", default="generated_sequence.png", help="Output image path")
    parser.add_argument("--weights", default=None, help="Optional generator weights path")
    args = parser.parse_args()

    main(args.sequence, args.out, args.weights)