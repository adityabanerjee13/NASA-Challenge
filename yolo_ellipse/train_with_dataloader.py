"""
Training script for YOLO OBB model using NASA_dataloader.

This script uses the NASAImageDataset to load data and converts it to
YOLO OBB format on-the-fly for training.

Usage:
    python train_with_dataloader.py --epochs 100 --imgsz 1024

The script will:
1. Load data using NASA_dataloader with internal train/val split
2. Generate YOLO OBB format labels dynamically
3. Train the YOLO OBB model
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from NASA_dataloader import NASAImageDataset, ellipse_to_obb_corners, IMG_WIDTH, IMG_HEIGHT
from ultralytics import YOLO


def prepare_yolo_dataset(
    root_dir: str = "./train",
    csv_path: str = "./train-gt.csv",
    output_dir: str = "./yolo_dataset",
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: int = None
):
    """
    Prepare YOLO OBB dataset using NASA_dataloader.

    Creates the directory structure required by Ultralytics YOLO:
        output_dir/
            images/train/
            images/val/
            labels/train/
            labels/val/
    """
    print("=" * 60)
    print("Preparing YOLO OBB Dataset from NASA_dataloader")
    print("=" * 60)

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Load datasets using NASA_dataloader
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

    stats = {'train': 0, 'val': 0, 'train_annotations': 0, 'val_annotations': 0, 'skipped': 0}

    # Process each split
    for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
        print(f"\nProcessing {split} split ({len(dataset)} images)...")

        for idx in range(len(dataset)):
            try:
                # Get image path and annotations
                img_path = dataset.get_image_path(idx)
                annotations = dataset.get_annotations(idx)
                img_key = dataset.image_paths[idx]

                # Create flat filename
                flat_name = img_key.replace('\\', '_').replace('/', '_')

                # Destination paths
                dst_img = output_path / 'images' / split / f"{flat_name}.png"
                dst_label = output_path / 'labels' / split / f"{flat_name}.txt"

                # Copy image (or create symlink)
                if not dst_img.exists():
                    src_img = Path(img_path)
                    if src_img.exists():
                        try:
                            # Try symlink first (faster, saves space)
                            dst_img.symlink_to(src_img.resolve())
                        except OSError:
                            # Fallback to copy on Windows
                            shutil.copy2(src_img, dst_img)
                    else:
                        print(f"Warning: Image not found: {src_img}")
                        stats['skipped'] += 1
                        continue

                # Generate label file
                label_lines = []
                for ann in annotations:
                    # Get class ID directly (classes are 0-4)
                    class_id = ann['crater_classification']

                    # Convert ellipse to OBB corners
                    corners = ellipse_to_obb_corners(
                        ann['ellipseCenterX(px)'],
                        ann['ellipseCenterY(px)'],
                        ann['ellipseSemimajor(px)'],
                        ann['ellipseSemiminor(px)'],
                        ann['ellipseRotation(deg)']
                    )

                    # Normalize and format
                    coords = []
                    for x, y in corners:
                        coords.extend([f"{x / IMG_WIDTH:.6f}", f"{y / IMG_HEIGHT:.6f}"])

                    label_lines.append(f"{class_id} " + " ".join(coords))
                    stats[f'{split}_annotations'] += 1

                # Write label file
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(label_lines))

                stats[split] += 1

                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{len(dataset)} images...")

            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                stats['skipped'] += 1

    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Train images: {stats['train']} ({stats['train_annotations']} annotations)")
    print(f"  Val images: {stats['val']} ({stats['val_annotations']} annotations)")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Output: {output_path.resolve()}")

    return output_path


def create_dataset_yaml(output_dir: str, yaml_path: str = None):
    """Create dataset.yaml configuration file."""
    output_path = Path(output_dir).resolve()

    if yaml_path is None:
        yaml_path = output_path / 'dataset.yaml'

    yaml_content = f"""# NASA Crater Detection Dataset - YOLO OBB Format
# Auto-generated by train_with_dataloader.py

path: {output_path}

train: images/train
val: images/val

# Class names (crater_classification 2,3,4 mapped to 0,1,2)
names:
  0: crater_class_0
  1: crater_class_1
  2: crater_class_2
  3: crater_class_3
  4: crater_class_4

nc: 5
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Dataset YAML created: {yaml_path}")
    return yaml_path


def train(
    data: str,
    model: str = 'yolov11l-obb.pt',
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 8,
    device: str = '',
    workers: int = 8,
    project: str = 'runs/train',
    name: str = 'crater_obb',
    resume: bool = False,
    patience: int = 50,
    lr0: float = 0.01,
):
    """Train YOLO OBB model."""
    print("\n" + "=" * 60)
    print("Starting YOLO OBB Training")
    print("=" * 60)

    # Load model
    if resume:
        model_path = Path(project) / name / 'weights' / 'last.pt'
        if model_path.exists():
            print(f"Resuming from: {model_path}")
            yolo_model = YOLO(str(model_path))
        else:
            print(f"No checkpoint found, starting fresh with {model}")
            yolo_model = YOLO(model)
    else:
        print(f"Loading model: {model}")
        yolo_model = YOLO(model)

    # Training arguments
    train_args = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'workers': workers,
        'cache': False,
        'project': project,
        'name': name,
        'patience': patience,
        'lr0': lr0,
        'deterministic': False,
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'save': True,
        'save_period': 1,
        # Augmentation for crater detection
        'degrees': 0,      # Full rotation (craters are rotation-invariant)
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.0,
        'scale': 0.0,
        'translate': 0.0,
    }

    if device:
        train_args['device'] = device

    print(f"\nConfiguration:")
    print(f"  Data: {data}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device if device else 'auto'}")
    print("=" * 60 + "\n")

    # Train
    results = yolo_model.train(**train_args)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model: {Path(project) / name / 'weights' / 'best.pt'}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO OBB model using NASA_dataloader'
    )

    # Data arguments
    parser.add_argument('--root-dir', default='./train', help='Training images directory')
    parser.add_argument('--csv', default='./train-gt.csv', help='Ground truth CSV')
    parser.add_argument('--output-dir', default='./yolo_dataset', help='YOLO dataset output')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (for testing)')

    # Training arguments
    parser.add_argument('--model', default='yolo11n-obb.pt',
                       choices=['yolo11n-obb.pt', 'yolo11s-obb.pt', 'yolo11m-obb.pt',
                               'yolo11l-obb.pt', 'yolo11x-obb.pt'],
                       help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', default='', help='Device (0, 0,1, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--project', default='./runs/train', help='Project directory')
    parser.add_argument('--name', default='crater_obb', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')

    # Workflow control
    parser.add_argument('--skip-prepare', action='store_true',
                       help='Skip dataset preparation (use existing)')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare dataset, skip training')

    args = parser.parse_args()

    # Step 1: Prepare dataset
    if not args.skip_prepare:
        output_path = prepare_yolo_dataset(
            root_dir=args.root_dir,
            csv_path=args.csv,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            max_samples=args.max_samples
        )

        yaml_path = create_dataset_yaml(args.output_dir)
    else:
        yaml_path = Path(args.output_dir) / 'dataset.yaml'
        if not yaml_path.exists():
            yaml_path = create_dataset_yaml(args.output_dir)

    if args.prepare_only:
        print("\nDataset preparation complete. Skipping training.")
        return

    # Step 2: Train model
    train(
        data=str(yaml_path),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        lr0=args.lr0,
    )


if __name__ == '__main__':
    main()
