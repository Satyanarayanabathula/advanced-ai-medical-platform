import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

from app.ml.dataset import create_datasets
from app.ml.gradcam import GradCAM
from app.ml.model import create_model


MODEL_PATH = "model/best_resnet18.pth"
OUTPUT_DIR = "results/gradcam"

CLASS_NAMES = {
    0: "Normal",
    1: "Pneumonia",
}


def denormalize(image):
    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image = image.transpose(1, 2, 0)

    image = image * std + mean

    image = np.clip(image, 0, 1)

    return image


def find_cases(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    cases = {
        "correct_normal": None,
        "correct_pneumonia": None,
        "false_positive": None,
        "false_negative": None,
    }

    model.eval()

    with torch.no_grad():

        for index, (image, label) in enumerate(loader):

            image_gpu = image.to(device)

            output = model(image_gpu)

            prediction = output.argmax(
                dim=1
            ).item()

            actual = label.item()

            if (
                actual == 0
                and prediction == 0
                and cases["correct_normal"] is None
            ):
                cases["correct_normal"] = index

            elif (
                actual == 1
                and prediction == 1
                and cases["correct_pneumonia"] is None
            ):
                cases["correct_pneumonia"] = index

            elif (
                actual == 0
                and prediction == 1
                and cases["false_positive"] is None
            ):
                cases["false_positive"] = index

            elif (
                actual == 1
                and prediction == 0
                and cases["false_negative"] is None
            ):
                cases["false_negative"] = index

            if all(
                value is not None
                for value in cases.values()
            ):
                break

    return cases


def generate_visualization(
    model,
    dataset,
    index,
    case_name,
    device,
):
    image, label = dataset[index]

    image_batch = image.unsqueeze(0).to(device)

    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    heatmap, output, predicted_class = (
        gradcam.generate(
            image_batch
        )
    )

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]

    heatmap = heatmap.cpu().numpy()

    original_image = (
        denormalize(
            image.numpy()
        )
    )

    actual_class = int(
    np.asarray(label).reshape(-1)[0]
    )

    confidence = probabilities[
        predicted_class
    ].item()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        original_image,
        cmap="gray",
    )

    axes[0].set_title(
        f"Original X-ray\n"
        f"Actual: {CLASS_NAMES[actual_class]}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        original_image,
        cmap="gray",
    )

    axes[1].imshow(
        heatmap,
        alpha=0.45,
        cmap="jet",
    )

    axes[1].set_title(
        f"Grad-CAM\n"
        f"Predicted: {CLASS_NAMES[predicted_class]}\n"
        f"Confidence: {confidence:.2%}"
    )

    axes[1].axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{case_name}.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    gradcam.close()

    print(
        f"Saved: {output_path}"
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    _, _, test_dataset = create_datasets()

    print(
        "Searching test set for representative cases..."
    )

    cases = find_cases(
        model,
        test_dataset,
        device,
    )

    print()
    print("Selected cases:")

    for case_name, index in cases.items():
        print(
            f"{case_name}: {index}"
        )

    print()

    for case_name, index in cases.items():

        if index is None:
            print(
                f"Could not find case: {case_name}"
            )
            continue

        generate_visualization(
            model=model,
            dataset=test_dataset,
            index=index,
            case_name=case_name,
            device=device,
        )

    print()
    print("Grad-CAM generation complete.")


if __name__ == "__main__":
    main()