from collections import Counter

from medmnist import PneumoniaMNIST


CLASS_NAMES = {
    0: "Normal",
    1: "Pneumonia",
}


def analyze_split(split):
    dataset = PneumoniaMNIST(
        split=split,
        download=True,
        size=224,
    )

    labels = [int(label[0]) for label in dataset.labels]

    counts = Counter(labels)

    print(f"\n{split.upper()} SET")
    print("-" * 30)
    print(f"Total images: {len(dataset)}")

    for class_id, class_name in CLASS_NAMES.items():
        count = counts[class_id]
        percentage = (count / len(dataset)) * 100

        print(
            f"{class_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


def main():
    print("PneumoniaMNIST Class Distribution")
    print("=" * 40)

    for split in ["train", "val", "test"]:
        analyze_split(split)


if __name__ == "__main__":
    main()