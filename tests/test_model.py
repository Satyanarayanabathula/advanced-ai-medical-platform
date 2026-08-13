import torch


from app.ml.model import create_model


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model()
    model = model.to(device)

    model.eval()

    print("Model device:", next(model.parameters()).device)

    print("Running model test...")

    # Create a fresh test batch
    from medmnist import PneumoniaMNIST
    from torch.utils.data import DataLoader
    from app.ml.preprocessing import eval_transform

    dataset = PneumoniaMNIST(
        split="train",
        transform=eval_transform,
        download=True,
        size=224,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(dataloader))

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Output device:", outputs.device)

    print("Model test successful!")


if __name__ == "__main__":
    main() 