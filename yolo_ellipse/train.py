"""
Training script for YOLO OBB model on NASA crater dataset.

Usage:
    python train.py --data dataset.yaml --epochs 100 --imgsz 1024

This script trains a YOLOv8 OBB (Oriented Bounding Box) model
for crater detection and classification.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data: str = 'dataset.yaml',
    model: str = 'yolov8n-obb.pt',
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 8,
    device: str = '',
    workers: int = 8,
    project: str = 'runs/train',
    name: str = 'crater_obb',
    resume: bool = False,
    pretrained: bool = True,
    patience: int = 50,
    save_period: int = 10,
    lr0: float = 0.01,
    lrf: float = 0.01,
    augment: bool = True,
):
    """
    Train YOLO OBB model for crater detection.

    Args:
        data: Path to dataset YAML configuration
        model: Pretrained model to start from (yolov8n-obb, yolov8s-obb, yolov8m-obb, etc.)
        epochs: Number of training epochs
        imgsz: Input image size (images will be resized)
        batch: Batch size
        device: Device to train on ('', '0', '0,1', 'cpu')
        workers: Number of dataloader workers
        project: Project directory for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights
        patience: Early stopping patience (epochs)
        save_period: Save checkpoint every N epochs
        lr0: Initial learning rate
        lrf: Final learning rate factor
        augment: Enable data augmentation
    """
    # Load model
    if resume:
        # Resume from last checkpoint
        model_path = Path(project) / name / 'weights' / 'last.pt'
        if model_path.exists():
            print(f"Resuming from {model_path}")
            yolo_model = YOLO(str(model_path))
        else:
            print(f"No checkpoint found at {model_path}, starting fresh")
            yolo_model = YOLO(model)
    else:
        print(f"Loading pretrained model: {model}")
        yolo_model = YOLO(model)

    # Training configuration
    train_args = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'workers': workers,
        'project': project,
        'name': name,
        'patience': patience,
        'save_period': save_period,
        'lr0': lr0,
        'lrf': lrf,
        'exist_ok': True,
        'pretrained': pretrained,
        'verbose': True,
        'plots': True,
    }

    # Add device if specified
    if device:
        train_args['device'] = device

    # Augmentation settings for crater detection
    if augment:
        train_args.update({
            'hsv_h': 0.015,     # HSV-Hue augmentation
            'hsv_s': 0.7,      # HSV-Saturation augmentation
            'hsv_v': 0.4,      # HSV-Value augmentation
            'degrees': 180,     # Rotation augmentation (craters can be any orientation)
            'translate': 0.1,  # Translation augmentation
            'scale': 0.5,      # Scale augmentation
            'shear': 0.0,      # Shear augmentation
            'perspective': 0.0,  # Perspective augmentation
            'flipud': 0.5,     # Vertical flip probability
            'fliplr': 0.5,     # Horizontal flip probability
            'mosaic': 1.0,     # Mosaic augmentation probability
            'mixup': 0.0,      # Mixup augmentation
        })

    print("\n" + "="*60)
    print("Starting YOLO OBB Training for Crater Detection")
    print("="*60)
    print(f"  Data config: {data}")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device if device else 'auto'}")
    print("="*60 + "\n")

    # Start training
    results = yolo_model.train(**train_args)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {Path(project) / name / 'weights' / 'last.pt'}")
    print("="*60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO OBB model for crater detection'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='dataset.yaml',
        help='Path to dataset YAML (default: dataset.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-obb.pt',
        choices=['yolov8n-obb.pt', 'yolov8s-obb.pt', 'yolov8m-obb.pt',
                 'yolov8l-obb.pt', 'yolov8x-obb.pt'],
        help='Pretrained model size (default: yolov8n-obb.pt)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: 100)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1024,
        help='Image size (default: 1024)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device (e.g., 0 or 0,1 or cpu)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Dataloader workers (default: 8)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory (default: runs/train)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='crater_obb',
        help='Experiment name (default: crater_obb)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )

    args = parser.parse_args()

    train(
        data=args.data,
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
        augment=not args.no_augment,
    )


if __name__ == '__main__':
    main()
