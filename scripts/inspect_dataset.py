from medmnist import PneumoniaMNIST


def main():
    dataset = PneumoniaMNIST(
        split="train",
        download=True
    )

    print("Dataset loaded successfully")
    print("Number of samples:", len(dataset))
    print("Number of classes:", len(dataset.info["label"]))

    print("\nDataset information:")
    print(dataset.info)

    image, label = dataset[0]

    print("\nFirst sample:")
    print("Image type:", type(image))
    print("Image size:", image.size)
    print("Label:", label)


if __name__ == "__main__":
    main()