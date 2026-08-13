from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def denormalize_tensor(image_tensor: torch.Tensor):
    image = image_tensor.detach().cpu().numpy()

    image = image.transpose(1, 2, 0)

    image = image * STD + MEAN

    image = np.clip(image, 0.0, 1.0)

    return image


def save_gradcam_overlay(
    original_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    output_path: Path,
):
    original_image = denormalize_tensor(
        original_tensor
    )

    heatmap_array = (
        heatmap.detach()
        .cpu()
        .numpy()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        original_image,
        cmap="gray",
    )
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(
        original_image,
        cmap="gray",
    )
    axes[1].imshow(
        heatmap_array,
        alpha=0.45,
        cmap="jet",
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)