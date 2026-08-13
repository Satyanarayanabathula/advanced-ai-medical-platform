import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from app.ml.dataset import create_datasets
from app.ml.model import create_model


MODEL_PATH = "model/best_resnet18.pth"
BATCH_SIZE = 16


def get_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            all_labels.extend(
                labels.reshape(-1).numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_probabilities),
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 60)

    _, val_dataset, _ = create_datasets()

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model()
    model = model.to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    labels, probabilities = get_predictions(
        model,
        val_loader,
        device,
    )

    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Specificity':<12}"
    )

    print("-" * 72)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        cm = confusion_matrix(
            labels,
            predictions,
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp)

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{specificity:<12.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print()
    print("=" * 60)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation F1: {best_f1:.4f}"
    )


if __name__ == "__main__":
    main()