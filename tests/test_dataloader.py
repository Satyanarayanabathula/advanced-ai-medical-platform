import torch
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST

from app.ml.preprocessing import train_transform


def main():
    dataset = PneumoniaMNIST(
        split="train",
        transform=train_transform,
        download=True,
        size=224,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(dataloader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    images = images.to(device)
    labels = labels.to(device)

    print("Device:", device)
    print("Images device:", images.device)
    print("Labels device:", labels.device)


if __name__ == "__main__":
    main()