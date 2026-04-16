"""
model.py
--------
FastViT model setup, weighted loss, optimiser, and the
full training + validation loop for the Industrial Defect Inspector.
"""

import torch
import torch.nn as nn
import timm
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Model 
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """
    Load FastViT-T8 (pre-trained on ImageNet) and replace the
    classification head to output *num_classes* logits.

    Returns:
        model   – the ready-to-train model on *device*
        device  – the torch.device being used
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("fastvit_t8", pretrained=pretrained)

    # Swap the final linear layer
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    print(f"FastViT-T8 ready on: {device}")
    return model, device


# ---------------------------------------------------------------------------
# Loss & optimiser
# ---------------------------------------------------------------------------

def get_criterion_and_optimizer(
    model: nn.Module,
    device: torch.device,
    class_weights: list[float] | None = None,
    lr: float = 1e-4,
) -> tuple[nn.CrossEntropyLoss, torch.optim.Optimizer]:
    """
    Build a weighted CrossEntropyLoss (to handle class imbalance)
    and an AdamW optimiser.

    Default weights [1.0, 3.6] up-weight the minority 'damaged' class.
    Pass *class_weights=None* for a standard unweighted loss.
    """
    if class_weights is None:
        class_weights = [1.0, 3.6]

    weights   = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    return criterion, optimizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch} Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Run validation. Returns accuracy as a percentage."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
) -> dict[str, list]:
    """
    Run the complete training & validation loop.

    Returns:
        history – dict with keys 'train_loss' and 'val_acc'
                  (one entry per epoch, useful for plotting).
    """
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_acc = validate(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:>2} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        print("-" * 50)

    print("Training complete — FastViT is now an Industrial Inspector.")
    return history
