import argparse
import os
from typing import Optional, Tuple

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


def main(number_string: str = "101", output_path: str = "generated_sequence.png"):
    generator, device = load_generator()
    sequence_image = generate_sequence(number_string, generator, device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(sequence_image, output_path)
    print(f"Saved generated sequence '{number_string}' to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence of MNIST digits with a trained cGAN")
    parser.add_argument(
        "sequence_positional",
        nargs="?",
        default=None,
        help="Digit string to generate, e.g. 101 or 90817",
    )
    parser.add_argument(
        "out_positional",
        nargs="?",
        default=None,
        help="Path to save output image",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Digit string to generate, e.g. 101 or 90817",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save output image",
    )
    args = parser.parse_args()

    sequence = args.sequence if args.sequence is not None else args.sequence_positional
    output_path = args.out if args.out is not None else args.out_positional

    if sequence is None:
        sequence = "101"
    if output_path is None:
        output_path = "generated_sequence.png"

    main(sequence, output_path)