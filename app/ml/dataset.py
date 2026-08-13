from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader

from app.ml.preprocessing import train_transform, eval_transform


def create_datasets():
    train_dataset = PneumoniaMNIST(
        split="train",
        transform=train_transform,
        download=True,
        size=224,
    )

    val_dataset = PneumoniaMNIST(
        split="val",
        transform=eval_transform,
        download=True,
        size=224,
    )

    test_dataset = PneumoniaMNIST(
        split="test",
        transform=eval_transform,
        download=True,
        size=224,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size=16):
    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader