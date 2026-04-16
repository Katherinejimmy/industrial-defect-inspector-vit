"""
data_preprocessing.py
---------------------
Handles raw MVTec data ingestion, train/val/test splitting, and
DataLoader creation for the Industrial Defect Inspector project.
"""

import os
import shutil
import random

import torch
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
TRAIN_PERC = 0.70
VAL_PERC   = 0.15
# TEST_PERC  = 1 - TRAIN_PERC - VAL_PERC  (implicitly 0.15)

# Standard transform — same for train, val, and test
DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def create_split_dirs(base_dir: str) -> None:
    """
    Create the folder skeleton:
        <base_dir>/
            train/good   train/damaged
            val/good     val/damaged
            test/good    test/damaged
    """
    for split in ("train", "val", "test"):
        for cls in ("good", "damaged"):
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
    print(f"Empty folder structure created at: {base_dir}")


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def gather_good_images(mvtec_root: str) -> list[str]:
    """
    Collect all .png paths labelled 'good' from both the
    official train/good and test/good directories.
    """
    good_paths = []
    for sub in ("train/good", "test/good"):
        folder = os.path.join(mvtec_root, sub)
        if os.path.isdir(folder):
            good_paths += [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".png")
            ]
    return good_paths


def gather_damaged_images(mvtec_root: str) -> list[str]:
    """
    Collect all .png paths from every sub-folder inside
    test/ that is NOT 'good'.
    """
    test_root = os.path.join(mvtec_root, "test")
    damaged_paths = []
    for folder in os.listdir(test_root):
        if folder == "good":
            continue
        folder_path = os.path.join(test_root, folder)
        if os.path.isdir(folder_path):
            damaged_paths += [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".png")
            ]
    return damaged_paths


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def get_split_files(
    file_list: list[str],
    train_perc: float = TRAIN_PERC,
    val_perc: float   = VAL_PERC,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Randomly shuffle then split a file list into train / val / test.
    Setting a fixed seed guarantees the same split every run.
    """
    random.seed(seed)
    shuffled = file_list[:]
    random.shuffle(shuffled)

    n          = len(shuffled)
    train_idx  = int(n * train_perc)
    val_idx    = int(n * (train_perc + val_perc))

    return shuffled[:train_idx], shuffled[train_idx:val_idx], shuffled[val_idx:]


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------

def _copy_files(file_list: list[str], destination: str) -> None:
    """Copy a list of files into *destination* directory."""
    for src in file_list:
        shutil.copy(src, destination)


def build_dataset(mvtec_root: str, base_dir: str) -> None:
    """
    Full pipeline: gather → split → copy into the
    train/val/test directory tree at *base_dir*.

    Args:
        mvtec_root: Root of the raw MVTec bottle data
                    (e.g. '/content/mvtec-ad/bottle').
        base_dir:   Destination root
                    (e.g. '/content/industrial_data').
    """
    create_split_dirs(base_dir)

    all_good    = gather_good_images(mvtec_root)
    all_damaged = gather_damaged_images(mvtec_root)

    print(f"Gathered: {len(all_good)} good | {len(all_damaged)} damaged")

    good_train,    good_val,    good_test    = get_split_files(all_good)
    damaged_train, damaged_val, damaged_test = get_split_files(all_damaged)

    # --- copy ---
    _copy_files(good_train,    os.path.join(base_dir, "train/good"))
    _copy_files(good_val,      os.path.join(base_dir, "val/good"))
    _copy_files(good_test,     os.path.join(base_dir, "test/good"))
    _copy_files(damaged_train, os.path.join(base_dir, "train/damaged"))
    _copy_files(damaged_val,   os.path.join(base_dir, "val/damaged"))
    _copy_files(damaged_test,  os.path.join(base_dir, "test/damaged"))

    print("Dataset ready.")
    print(f"  Train  — good: {len(good_train)} | damaged: {len(damaged_train)}")
    print(f"  Val    — good: {len(good_val)}   | damaged: {len(damaged_val)}")
    print(f"  Test   — good: {len(good_test)}  | damaged: {len(damaged_test)}")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[torch.utils.data.DataLoader, ...]:
    """
    Return (train_loader, val_loader, test_loader, train_dataset).

    *train_dataset* is exposed so callers can read .classes, etc.
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(base_dir, "train"), transform=DATA_TRANSFORM
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(base_dir, "val"), transform=DATA_TRANSFORM
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(base_dir, "test"), transform=DATA_TRANSFORM
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset
