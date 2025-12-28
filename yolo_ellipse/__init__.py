"""
YOLO OBB-based Crater Ellipse Detection Package

This package provides tools for training and running inference with
YOLO OBB models for crater detection, with conversion to/from ellipse parameters.

Modules:
    - convert_to_yolo_obb: Convert NASA crater CSV to YOLO OBB format
    - train: Train YOLO OBB models
    - predict: Run inference and convert OBB to ellipse predictions

Usage:
    # Convert dataset
    python -m yolo_ellipse.convert_to_yolo_obb --csv train-gt.csv --images train/ --output yolo_dataset/

    # Train model
    python -m yolo_ellipse.train --data yolo_ellipse/dataset.yaml --epochs 100

    # Run predictions
    python -m yolo_ellipse.predict --model runs/train/crater_obb/weights/best.pt --source test/ --output predictions.csv
"""

from .predict import CraterPredictor, EllipsePrediction, save_predictions_csv
from .convert_to_yolo_obb import ellipse_to_obb_corners, convert_dataset

__version__ = '1.0.0'
__all__ = [
    'CraterPredictor',
    'EllipsePrediction',
    'save_predictions_csv',
    'ellipse_to_obb_corners',
    'convert_dataset',
]
