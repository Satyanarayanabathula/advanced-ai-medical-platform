from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt


def main():
    dataset = PneumoniaMNIST(
        split="train",
        download=True,
        size=224
    )

    class_names = {
        0: "Normal",
        1: "Pneumonia"
    }

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]

        ax.imshow(image, cmap="gray")
        ax.set_title(class_names[int(label[0])])
        ax.axis("off")

    plt.tight_layout()

    output_path = "results/dataset_samples_224.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()