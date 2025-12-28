import os
import glob
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


# Image dimensions
IMG_WIDTH = 2592
IMG_HEIGHT = 2048


class NASAImageDataset(Dataset):
    """
    Dataset for NASA challenge images with train/val split support.

    Args:
        root_dir: Path to the train/ directory containing images
        csv_path: Path to train-gt.csv
        transform: Optional torchvision transforms
        split: 'train', 'val', or 'all' (default: 'all')
        train_ratio: Ratio for train/val split (default: 0.8)
        seed: Random seed for reproducible splits (default: 42)
        max_samples: Maximum samples to load (None for all)
    """

    def __init__(
        self,
        root_dir="./train",
        csv_path="./train-gt.csv",
        transform=None,
        split='all',
        train_ratio=0.8,
        seed=42,
        max_samples=None
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self.split = split

        # Parse CSV and group by image
        df = pd.read_csv(csv_path)

        self.file_set = dict()

        for i, row in df.iterrows():
            img = row['inputImage']

            if img not in self.file_set:
                self.file_set[img] = []

            self.file_set[img].append({
                'ellipseCenterX(px)': row['ellipseCenterX(px)'],
                'ellipseCenterY(px)': row['ellipseCenterY(px)'],
                'ellipseSemimajor(px)': row['ellipseSemimajor(px)'],
                'ellipseSemiminor(px)': row['ellipseSemiminor(px)'],
                'ellipseRotation(deg)': row['ellipseRotation(deg)'],
                'crater_classification': row['crater_classification']
            })

        # Get all image paths and apply split
        all_image_paths = list(self.file_set.keys())

        # Shuffle with fixed seed for reproducibility
        random.seed(seed)
        random.shuffle(all_image_paths)

        # Apply train/val split
        split_idx = int(len(all_image_paths) * train_ratio)

        if split == 'train':
            self.image_paths = all_image_paths[:split_idx]
        elif split == 'val':
            self.image_paths = all_image_paths[split_idx:]
        else:  # 'all'
            self.image_paths = all_image_paths

        # Apply max_samples limit if specified
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        print(f"[{split.upper()}] Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def _parse_image_path(self, img_key):
        """Parse image key to construct file paths."""
        # Handle both forward and backward slashes
        parts = img_key.replace('/', '\\').split('\\')
        if len(parts) >= 3:
            altitude, longitude, orientation_light = parts[0], parts[1], parts[2]
        else:
            # Fallback for unexpected format
            parts = img_key.replace('\\', '/').split('/')
            altitude, longitude, orientation_light = parts[0], parts[1], parts[2]

        img_path = os.path.join(self.root_dir, altitude, longitude, orientation_light) + '.png'
        mask_path = os.path.join(self.root_dir, altitude, longitude, 'truth', orientation_light + '_mask.png')
        truth_path = os.path.join(self.root_dir, altitude, longitude, 'truth', orientation_light + '_truth.png')

        return img_path, mask_path, truth_path

    def __getitem__(self, idx):
        img_key = self.image_paths[idx]
        img_path, mask_path, truth_path = self._parse_image_path(img_key)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.file_set[img_key], img_key

    def get_annotations(self, idx):
        """Get annotations for a specific index."""
        img_key = self.image_paths[idx]
        return self.file_set[img_key]

    def get_image_path(self, idx):
        """Get the full image path for a specific index."""
        img_key = self.image_paths[idx]
        img_path, _, _ = self._parse_image_path(img_key)
        return img_path


def ellipse_to_obb_corners(cx, cy, semimajor, semiminor, rotation_deg):
    """
    Convert ellipse parameters to 4 OBB corner points.

    Returns:
        List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    half_w = semimajor
    half_h = semiminor

    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners_local = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    corners = []
    for dx, dy in corners_local:
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        corners.append((cx + rx, cy + ry))

    return corners


class NASAYOLODataset(Dataset):
    """
    YOLO OBB compatible dataset wrapper for NASA crater data.

    This dataset returns data in the format expected by Ultralytics YOLO OBB training.

    Args:
        root_dir: Path to the train/ directory
        csv_path: Path to train-gt.csv
        split: 'train' or 'val'
        train_ratio: Train/val split ratio
        imgsz: Target image size for resizing
        seed: Random seed for split
    """

    def __init__(
        self,
        root_dir="./train",
        csv_path="./train-gt.csv",
        split='train',
        train_ratio=0.8,
        imgsz=1024,
        seed=42,
        max_samples=None
    ):
        self.base_dataset = NASAImageDataset(
            root_dir=root_dir,
            csv_path=csv_path,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            max_samples=max_samples
        )
        self.imgsz = imgsz

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image or numpy array
            labels: numpy array of shape (N, 9) where each row is:
                    [class_id, x1, y1, x2, y2, x3, y3, x4, y4] (normalized)
            img_path: Original image path key
        """
        image, annotations, img_key = self.base_dataset[idx]

        # Convert annotations to YOLO OBB format
        labels = []
        for ann in annotations:
            # Get class ID (map 2,3,4 -> 0,1,2)
            cls = ann['crater_classification']
            class_id = cls

            # Convert ellipse to OBB corners
            corners = ellipse_to_obb_corners(
                ann['ellipseCenterX(px)'],
                ann['ellipseCenterY(px)'],
                ann['ellipseSemimajor(px)'],
                ann['ellipseSemiminor(px)'],
                ann['ellipseRotation(deg)']
            )

            # Normalize coordinates
            norm_corners = []
            for x, y in corners:
                norm_corners.extend([x / IMG_WIDTH, y / IMG_HEIGHT])

            # Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
            labels.append([class_id] + norm_corners)

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 9), dtype=np.float32)

        return image, labels, img_key

    def get_image_path(self, idx):
        """Get full image file path."""
        return self.base_dataset.get_image_path(idx)


def get_train_val_datasets(
    root_dir="./train",
    csv_path="./train-gt.csv",
    train_ratio=0.8,
    seed=42,
    max_samples=None
):
    """
    Convenience function to get train and val datasets.

    Returns:
        train_dataset, val_dataset: NASAImageDataset instances
    """
    train_dataset = NASAImageDataset(
        root_dir=root_dir,
        csv_path=csv_path,
        split='train',
        train_ratio=train_ratio,
        seed=seed,
        max_samples=max_samples
    )

    val_dataset = NASAImageDataset(
        root_dir=root_dir,
        csv_path=csv_path,
        split='val',
        train_ratio=train_ratio,
        seed=seed,
        max_samples=max_samples
    )

    return train_dataset, val_dataset


def get_yolo_datasets(
    root_dir="./train",
    csv_path="./train-gt.csv",
    train_ratio=0.8,
    imgsz=1024,
    seed=42,
    max_samples=None
):
    """
    Get train and val datasets in YOLO OBB format.

    Returns:
        train_dataset, val_dataset: NASAYOLODataset instances
    """
    train_dataset = NASAYOLODataset(
        root_dir=root_dir,
        csv_path=csv_path,
        split='train',
        train_ratio=train_ratio,
        imgsz=imgsz,
        seed=seed,
        max_samples=max_samples
    )

    val_dataset = NASAYOLODataset(
        root_dir=root_dir,
        csv_path=csv_path,
        split='val',
        train_ratio=train_ratio,
        imgsz=imgsz,
        seed=seed,
        max_samples=max_samples
    )

    return train_dataset, val_dataset
