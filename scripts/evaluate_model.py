import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from app.ml.dataset import create_datasets
from app.ml.model import create_model


MODEL_PATH = "model/best_resnet18.pth"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------
    # Load test dataset
    # ---------------------------------------------------------

    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    print("Test samples:", len(test_dataset))

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

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

    print("Best validation F1:", checkpoint["best_val_f1"])
    print("Best epoch:", checkpoint["epoch"])
    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.reshape(-1).long()

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

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

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TEST METRICS")
    print("=" * 60)

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {recall:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "Normal",
                "Pneumonia",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save confusion matrix
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Pneumonia"]
    )

    ax.set_yticklabels(
        ["Normal", "Pneumonia"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png",
    )

    plt.savefig(
        confusion_path,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()

    plt.tight_layout()

    roc_path = os.path.join(
        RESULTS_DIR,
        "roc_curve.png",
    )

    plt.savefig(
        roc_path,
        dpi=200,
    )

    plt.close()

    print()
    print("Saved:")
    print(confusion_path)
    print(roc_path)


if __name__ == "__main__":
    main()