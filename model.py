import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, img_dim: int = 28 * 28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = latent_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_emb(labels)
        gen_input = torch.cat((noise, label_vec), dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10, img_dim: int = 28 * 28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = img_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_img = img.view(img.size(0), -1)
        label_vec = self.label_emb(labels)
        disc_input = torch.cat((flat_img, label_vec), dim=1)
        return self.model(disc_input)