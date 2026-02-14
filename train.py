import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Discriminator, Generator


def train(
    epochs: int = 20,
    batch_size: int = 128,
    latent_dim: int = 100,
    lr: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(epochs):
        for batch_idx, (real_imgs, labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size_current = real_imgs.size(0)

            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            optimizer_d.zero_grad()
            real_logits = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_logits, valid)

            z = torch.randn(batch_size_current, latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (batch_size_current,), device=device)
            generated_imgs = generator(z, gen_labels)
            fake_logits = discriminator(generated_imgs.detach(), gen_labels)
            d_fake_loss = criterion(fake_logits, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_logits_for_g = discriminator(generated_imgs, gen_labels)
            g_loss = criterion(fake_logits_for_g, valid)
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}"
                )

    torch.save(generator.state_dict(), "checkpoints/generator.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator.pth")
    print("Training complete. Saved weights to checkpoints/.")


if __name__ == "__main__":
    train(epochs=20)