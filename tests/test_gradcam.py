import torch

from app.ml.model import create_model
from app.ml.gradcam import GradCAM
from app.ml.dataset import create_datasets
from torch.utils.data import DataLoader


MODEL_PATH = "model/best_resnet18.pth"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("GRAD-CAM TEST")
    print("=" * 60)

    model = create_model()
    model = model.to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded.")
    print("Device:", device)

    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    image, label = next(iter(test_loader))

    image = image.to(device)

    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    heatmap, output, predicted_class = gradcam.generate(
        image
    )

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    print()
    print("Actual class:", label.item())
    print("Predicted class:", predicted_class)
    print(
        "Normal probability:",
        probabilities[0, 0].item(),
    )
    print(
        "Pneumonia probability:",
        probabilities[0, 1].item(),
    )

    print("Heatmap shape:", heatmap.shape)
    print("Heatmap minimum:", heatmap.min().item())
    print("Heatmap maximum:", heatmap.max().item())

    gradcam.close()

    print()
    print("Grad-CAM computation successful!")


if __name__ == "__main__":
    main()