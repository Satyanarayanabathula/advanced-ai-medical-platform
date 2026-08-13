from pathlib import Path

from medmnist import PneumoniaMNIST

from app.ml.prediction_service import PredictionService


def main():

    print("=" * 60)
    print("PREDICTION SERVICE TEST")
    print("=" * 60)

    dataset = PneumoniaMNIST(
        split="test",
        download=True,
        size=224,
    )

    image, label = dataset[0]

    print("Actual label:", int(label[0]))
    print("Image size:", image.size)

    service = PredictionService()

    result = service.analyze(image)

    print()
    print("Prediction:", result["prediction"])

    print(
        "Probability:",
        f"{result['probability']:.4f}",
    )

    print()
    print("Class probabilities:")

    print(
        "Normal:",
        f"{result['probabilities']['normal']:.4f}",
    )

    print(
        "Pneumonia:",
        f"{result['probabilities']['pneumonia']:.4f}",
    )

    print()
    print(
        "Grad-CAM shape:",
        result["gradcam"].shape,
    )

    print()
    print("LLM Report:")
    print("-" * 60)

    print(result["report"])

    print()
    print(
        "Prediction service test successful!"
    )


if __name__ == "__main__":
    main()