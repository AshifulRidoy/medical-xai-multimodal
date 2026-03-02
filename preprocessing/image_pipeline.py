"""
preprocessing/image_pipeline.py

Purpose:
    PyTorch Dataset and DataLoader factory for chest X-ray images.
    Lazily loads images from manifest.csv, applies split-appropriate transforms,
    and returns (image_tensor, label_tensor, study_id) tuples.

Inputs:
    - data/manifest.csv (produced by build_manifest.py)
    - config.yaml

Outputs:
    - ChestXrayDataset instances and DataLoaders for each split

Example usage:
    from preprocessing.image_pipeline import get_dataloader
    import yaml
    config = yaml.safe_load(open("config.yaml"))
    train_loader = get_dataloader("train", config)
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Label classes (canonical ordering)
# ──────────────────────────────────────────────
LABEL_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# ImageNet normalization constants (required by AI_rules.md Section 3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return torchvision transforms for a given data split.

    Train split includes augmentation (flip, rotation, color jitter).
    Val and test splits use only resize and normalization.

    Args:
        split: One of "train", "val", "test"
        image_size: Target square image size in pixels (default 224)

    Returns:
        transforms.Compose pipeline
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # val and test: deterministic pipeline
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


class ChestXrayDataset(Dataset):
    """Lazy-loading PyTorch Dataset for MIMIC-CXR chest X-ray images.

    Loads images on demand (no full dataset in memory).
    Returns (image_tensor, label_tensor, study_id) for each sample.
    Skips samples with missing or unreadable images (logs a warning).

    Args:
        manifest_path: Path to data/manifest.csv
        split: One of "train", "val", "test"
        transform: torchvision transforms.Compose to apply
        label_classes: Ordered list of pathology label column names
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        label_classes: Optional[list] = None,
    ) -> None:
        """Initialize dataset from manifest CSV filtered to the given split."""
        assert os.path.exists(manifest_path), (
            f"Manifest not found: {manifest_path}. "
            "Run preprocessing/build_manifest.py first."
        )
        assert split in ("train", "val", "test"), f"Invalid split: {split}. Must be train/val/test."

        self.label_classes = label_classes or LABEL_CLASSES
        self.transform = transform or get_transforms(split)
        self._skipped: list = []

        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        assert len(self.data) > 0, f"No samples found for split='{split}' in {manifest_path}"
        logger.info(f"ChestXrayDataset [{split}]: {len(self.data)} samples")

    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Load and return one sample.

        Args:
            idx: Integer index into the dataset

        Returns:
            Tuple of (image_tensor [3, H, W], label_tensor [14], study_id str)
            Returns a zero-image tensor if the image file is missing.
        """
        import torch

        row = self.data.iloc[idx]
        study_id = str(row["study_id"])
        image_path = row["image_path"]

        # Load image
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            image_tensor = self.transform(image)
        except (FileNotFoundError, OSError) as e:
            if study_id not in self._skipped:
                logger.warning(f"Cannot load image for study {study_id}: {e}. Returning zeros.")
                self._skipped.append(study_id)
            # Return zero tensor of expected shape to avoid DataLoader crash
            image_tensor = torch.zeros(3, 224, 224)

        # Load labels
        label_values = row[self.label_classes].values.astype("float32")
        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        return image_tensor, label_tensor, study_id

    def get_label_counts(self) -> pd.Series:
        """Return per-class positive sample counts for this split.

        Returns:
            pandas Series with label class names as index
        """
        return self.data[self.label_classes].sum().astype(int)


def get_dataloader(split: str, config: dict, num_workers: int = 0) -> DataLoader:
    """Factory function that creates a DataLoader for a given split.

    Args:
        split: One of "train", "val", "test"
        config: Raw config dict loaded from config.yaml (yaml.safe_load)
        num_workers: Number of worker processes for data loading.
                     Default 0 (single-process) for MPS compatibility.

    Returns:
        torch.utils.data.DataLoader instance
    """
    image_size = config["dataset"]["image_size"]
    manifest_path = config["paths"]["manifest"]
    batch_size = config["cnn"]["batch_size"] if split == "train" else config["cnn"]["batch_size"] * 2

    transform = get_transforms(split, image_size)
    dataset = ChestXrayDataset(manifest_path, split, transform=transform)

    shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # pin_memory not supported on MPS
        drop_last=split == "train",  # avoid incomplete batches during training
    )
    logger.info(
        f"DataLoader [{split}]: {len(dataset)} samples, "
        f"batch_size={batch_size}, {len(loader)} batches"
    )
    return loader


def get_all_dataloaders(config: dict, num_workers: int = 0) -> dict:
    """Convenience function to create all three split DataLoaders.

    Args:
        config: Raw config dict from config.yaml
        num_workers: Workers per loader

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoader instances
    """
    return {
        split: get_dataloader(split, config, num_workers)
        for split in ("train", "val", "test")
    }


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    import yaml

    logging.basicConfig(level=logging.INFO)

    config = yaml.safe_load(open("config.yaml"))
    manifest_path = config["paths"]["manifest"]

    if not os.path.exists(manifest_path):
        print("Manifest not found. Run preprocessing/build_manifest.py first.")
    else:
        for split in ("train", "val", "test"):
            loader = get_dataloader(split, config)
            images, labels, ids = next(iter(loader))
            print(
                f"[{split}] images: {images.shape}, "
                f"labels: {labels.shape}, "
                f"study_ids[:2]: {ids[:2]}"
            )
        print("✓ image_pipeline smoke-test passed.")
