"""
Convert NASA crater dataset (train-gt.csv) to YOLO OBB format.

YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
The 4 corner points define the oriented bounding box.

Ellipse to OBB conversion:
- OBB is the bounding rectangle of the ellipse
- Width = 2 * semimajor, Height = 2 * semiminor
- Rotation applied to get the 4 corners
"""

import os
import csv
import math
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict


# Image dimensions
IMG_WIDTH = 2592
IMG_HEIGHT = 2048

# No class mapping needed - classes are already 0-4


def ellipse_to_obb_corners(
    cx: float, cy: float,
    semimajor: float, semiminor: float,
    rotation_deg: float
) -> List[Tuple[float, float]]:
    """
    Convert ellipse parameters to 4 OBB corner points.

    The OBB is the bounding rectangle of the ellipse, rotated by the ellipse rotation.
    For inscribed ellipse conversion (OBB -> ellipse), the semimajor/semiminor
    become half the width/height of the OBB.

    Args:
        cx, cy: Ellipse center coordinates (pixels)
        semimajor: Semi-major axis length (pixels)
        semiminor: Semi-minor axis length (pixels)
        rotation_deg: Rotation angle in degrees

    Returns:
        List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        in clockwise order starting from top-left (before rotation)
    """
    # OBB half-dimensions (width along semimajor, height along semiminor)
    half_w = semimajor  # half-width
    half_h = semiminor  # half-height

    # Convert rotation to radians
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Corner offsets before rotation (relative to center)
    # Starting from top-left, going clockwise
    corners_local = [
        (-half_w, -half_h),  # top-left
        (half_w, -half_h),   # top-right
        (half_w, half_h),    # bottom-right
        (-half_w, half_h),   # bottom-left
    ]

    # Apply rotation and translate to center
    corners = []
    for dx, dy in corners_local:
        # Rotate point
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        # Translate to center
        corners.append((cx + rx, cy + ry))

    return corners


def normalize_corners(
    corners: List[Tuple[float, float]],
    img_width: int,
    img_height: int
) -> List[Tuple[float, float]]:
    """Normalize corner coordinates to 0-1 range."""
    return [(x / img_width, y / img_height) for x, y in corners]


def format_yolo_obb_line(class_id: int, corners: List[Tuple[float, float]]) -> str:
    """
    Format a single YOLO OBB annotation line.

    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    """
    coords = []
    for x, y in corners:
        coords.extend([f"{x:.6f}", f"{y:.6f}"])
    return f"{class_id} " + " ".join(coords)


def parse_csv(csv_path: str) -> Dict[str, List[dict]]:
    """
    Parse train-gt.csv and group annotations by image.

    Returns:
        Dictionary mapping image_path -> list of crater annotations
    """
    annotations = defaultdict(list)

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['inputImage'].replace('\\', '/')  # Normalize path separators
            annotations[img_path].append({
                'cx': float(row['ellipseCenterX(px)']),
                'cy': float(row['ellipseCenterY(px)']),
                'semimajor': float(row['ellipseSemimajor(px)']),
                'semiminor': float(row['ellipseSemiminor(px)']),
                'rotation': float(row['ellipseRotation(deg)']),
                'classification': int(row['crater_classification'])
            })

    return annotations


def convert_dataset(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    copy_images: bool = False
):
    """
    Convert the entire dataset to YOLO OBB format.

    Creates:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/

    Args:
        csv_path: Path to train-gt.csv
        images_dir: Path to train/ directory containing images
        output_dir: Output directory for YOLO dataset
        train_ratio: Ratio of images for training (rest for validation)
        copy_images: If True, copy images; if False, create symlinks
    """
    print(f"Parsing annotations from {csv_path}...")
    annotations = parse_csv(csv_path)
    print(f"Found {len(annotations)} unique images with annotations")

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Split images into train/val
    image_paths = list(annotations.keys())
    split_idx = int(len(image_paths) * train_ratio)

    # Shuffle for randomness (using a fixed seed for reproducibility)
    import random
    random.seed(42)
    random.shuffle(image_paths)

    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    stats = {'train': 0, 'val': 0, 'total_annotations': 0, 'skipped': 0}

    for split, image_list in [('train', train_images), ('val', val_images)]:
        for img_rel_path in image_list:
            # Source image path
            src_img = Path(images_dir) / (img_rel_path + '.png')

            if not src_img.exists():
                print(f"Warning: Image not found: {src_img}")
                stats['skipped'] += 1
                continue

            # Create unique filename (flatten directory structure)
            flat_name = img_rel_path.replace('/', '_')

            # Destination paths
            dst_img = output_path / 'images' / split / f"{flat_name}.png"
            dst_label = output_path / 'labels' / split / f"{flat_name}.txt"

            # Copy or link image
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                # Create symlink (relative path for portability)
                if not dst_img.exists():
                    try:
                        dst_img.symlink_to(src_img.resolve())
                    except OSError:
                        # Fallback to copy if symlinks not supported (Windows)
                        shutil.copy2(src_img, dst_img)

            # Generate label file
            label_lines = []
            for ann in annotations[img_rel_path]:
                # Get class ID (use directly, classes are 0-4)
                class_id = ann['classification']

                # Convert ellipse to OBB corners
                corners = ellipse_to_obb_corners(
                    ann['cx'], ann['cy'],
                    ann['semimajor'], ann['semiminor'],
                    ann['rotation']
                )

                # Normalize coordinates
                norm_corners = normalize_corners(corners, IMG_WIDTH, IMG_HEIGHT)

                # Format line
                label_lines.append(format_yolo_obb_line(class_id, norm_corners))
                stats['total_annotations'] += 1

            # Write label file
            with open(dst_label, 'w') as f:
                f.write('\n'.join(label_lines))

            stats[split] += 1

    print(f"\nConversion complete!")
    print(f"  Train images: {stats['train']}")
    print(f"  Val images: {stats['val']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Skipped (not found): {stats['skipped']}")
    print(f"\nOutput directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NASA crater dataset to YOLO OBB format'
    )
    parser.add_argument(
        '--csv',
        default='./train-gt.csv',
        help='Path to train-gt.csv (default: train-gt.csv)'
    )
    parser.add_argument(
        '--images',
        default='./train',
        help='Path to train/ directory with images (default: train)'
    )
    parser.add_argument(
        '--output',
        default='./yolo_dataset',
        help='Output directory (default: yolo_dataset)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--copy-images',
        action='store_false',
        help='Copy images instead of creating symlinks'
    )

    args = parser.parse_args()

    convert_dataset(
        csv_path=args.csv,
        images_dir=args.images,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        copy_images=args.copy_images
    )


if __name__ == '__main__':
    main()
