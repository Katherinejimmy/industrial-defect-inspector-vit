"""
utils.py
--------
Utility functions for the Industrial Defect Inspector:
  - save / load model weights
  - single-image inference with visualisation
  - confusion matrix & classification report
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

from data_preprocessing import DATA_TRANSFORM


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, path: str) -> None:
    """Save model state-dict to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    Load weights from *path* into an already-constructed *model*
    and move it to *device*.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Weights loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def predict_image(
    model: nn.Module,
    image_path: str,
    class_names: list[str],
    device: torch.device,
    show: bool = True,
) -> tuple[str, float]:
    """
    Run inference on a single image file.

    Args:
        model       – trained FastViT model (eval mode)
        image_path  – path to the .png / .jpg file
        class_names – ordered list from train_dataset.classes
        device      – torch device
        show        – if True, display the image with its prediction

    Returns:
        (predicted_class, confidence_pct)
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = DATA_TRANSFORM(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output        = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label      = class_names[predicted.item()]
    confidence = confidence.item() * 100.0

    if show:
        plt.imshow(img)
        plt.title(f"Prediction: {label}  ({confidence:.2f}%)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return label, confidence


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    """
    Print a full classification report and plot a confusion matrix
    using the given data loader.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Text report
    print(classification_report(all_labels, all_preds, target_names=class_names))
