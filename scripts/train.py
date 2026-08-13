import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW

from app.ml.dataset import create_dataloaders
from app.ml.model import create_model


BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MODEL_DIR = "model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_resnet18.pth")


def calculate_class_weights(dataset):
    labels = np.asarray(dataset.labels).reshape(-1)

    class_counts = np.bincount(labels, minlength=2)
    total_samples = len(labels)
    num_classes = len(class_counts)

    weights = total_samples / (
        num_classes * class_counts
    )

    return torch.tensor(weights, dtype=torch.float32)


def calculate_metrics(labels, predictions, probabilities):
    accuracy = accuracy_score(labels, predictions)

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device)

        labels = labels.reshape(-1).long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[:, 1]

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    return epoch_loss, metrics


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            labels = labels.reshape(-1).long().to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    return epoch_loss, metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PNEUMONIA DETECTION - RESNET18 TRAINING")
    print("=" * 60)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    print()
    print("Class weights:")
    print("Normal:", class_weights[0].item())
    print("Pneumonia:", class_weights[1].item())

    class_weights = class_weights.to(device)

    model = create_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0

    print()
    print("Starting training...")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        print("-" * 60)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_f1,
                    "epoch": epoch,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"Best model saved to: "
                f"{BEST_MODEL_PATH}"
            )
            print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("Best validation F1:", best_f1)

    print()
    print("Loading best model...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print("FINAL TEST RESULTS")
    print("-" * 60)

    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['auc']:.4f}")

    print()
    print("Best model:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()