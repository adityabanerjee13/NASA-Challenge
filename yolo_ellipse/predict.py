"""
Inference script for predicting ellipses using YOLO OBB model.

This script:
1. Loads a trained YOLO OBB model
2. Runs inference on images
3. Converts OBB predictions back to ellipse parameters
4. Outputs results in CSV format matching the competition format

Usage:
    python predict.py --model runs/train/crater_obb/weights/best.pt --source test/ --output predictions.csv
"""

import os
import csv
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


# Image dimensions (used for denormalization)
IMG_WIDTH = 2592
IMG_HEIGHT = 2048

# No class mapping needed - classes are already 0-4


@dataclass
class EllipsePrediction:
    """Represents a predicted crater ellipse."""
    center_x: float
    center_y: float
    semimajor: float
    semiminor: float
    rotation_deg: float
    classification: int
    confidence: float
    image_path: str


def obb_to_ellipse(
    obb_points: np.ndarray,
    class_id: int,
    confidence: float,
    image_path: str
) -> EllipsePrediction:
    """
    Convert OBB (4 corner points) to ellipse parameters.

    For inscribed ellipse conversion:
    - Ellipse center = OBB center
    - Semimajor = half of OBB width
    - Semiminor = half of OBB height
    - Rotation = OBB rotation angle

    Args:
        obb_points: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        class_id: YOLO class ID (0, 1, 2, 3, or 4)
        confidence: Detection confidence
        image_path: Source image path

    Returns:
        EllipsePrediction with ellipse parameters
    """
    # Ensure we have 4 points
    points = np.array(obb_points).reshape(4, 2)

    # Calculate center (average of all corners)
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    # Calculate the two edge vectors from first point
    # Points are ordered: p0 -> p1 -> p2 -> p3 (clockwise or counter-clockwise)
    edge1 = points[1] - points[0]  # First edge
    edge2 = points[3] - points[0]  # Adjacent edge

    # Calculate edge lengths
    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)

    # Determine which edge is semimajor (longer) and which is semiminor
    if len1 >= len2:
        semimajor = len1 / 2
        semiminor = len2 / 2
        # Rotation angle from the longer edge
        rotation_rad = math.atan2(edge1[1], edge1[0])
    else:
        semimajor = len2 / 2
        semiminor = len1 / 2
        # Rotation angle from the longer edge
        rotation_rad = math.atan2(edge2[1], edge2[0])

    # Convert to degrees
    rotation_deg = math.degrees(rotation_rad)

    # Normalize rotation to 0-360 range (matching dataset format)
    rotation_deg = rotation_deg % 360
    if rotation_deg < 0:
        rotation_deg += 360

    # Use class ID directly (classes are 0-4)
    classification = class_id

    return EllipsePrediction(
        center_x=center_x,
        center_y=center_y,
        semimajor=semimajor,
        semiminor=semiminor,
        rotation_deg=rotation_deg,
        classification=classification,
        confidence=confidence,
        image_path=image_path
    )


def get_image_relative_path(image_path: str, base_dir: str) -> str:
    """
    Convert absolute image path to relative path format used in CSV.

    Example:
        Input: /path/to/train/altitude01/longitude02/orientation01_light01.png
        Output: altitude01/longitude02/orientation01_light01
    """
    # Get path relative to base directory
    rel_path = os.path.relpath(image_path, base_dir)

    # Remove .png extension
    if rel_path.endswith('.png'):
        rel_path = rel_path[:-4]

    # Normalize path separators
    rel_path = rel_path.replace('\\', '/')

    # Handle flattened filenames (altitude01_longitude02_orientation01_light01)
    if '_' in Path(rel_path).stem and '/' not in rel_path:
        # Reconstruct the original path format
        parts = Path(rel_path).stem.split('_')
        if len(parts) >= 4 and parts[0].startswith('altitude'):
            # Format: altitude##_longitude##_orientation##_light##
            altitude = parts[0]
            longitude = parts[1]
            orientation_light = '_'.join(parts[2:])
            rel_path = f"{altitude}/{longitude}/{orientation_light}"

    return rel_path


class CraterPredictor:
    """YOLO OBB-based crater ellipse predictor."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 100,
        device: str = ''
    ):
        """
        Initialize the crater predictor.

        Args:
            model_path: Path to trained YOLO OBB model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections per image
            device: Device to run inference on
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device

    def predict_image(
        self,
        image_path: str,
        base_dir: str = ''
    ) -> List[EllipsePrediction]:
        """
        Run prediction on a single image.

        Args:
            image_path: Path to input image
            base_dir: Base directory for relative path calculation

        Returns:
            List of EllipsePrediction objects
        """
        # Run YOLO inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device if self.device else None,
            verbose=False
        )

        predictions = []

        # Get relative path for CSV
        rel_path = get_image_relative_path(image_path, base_dir)

        # Process results
        for result in results:
            if result.obb is None:
                continue

            # Get OBB data
            obb_data = result.obb

            # obb_data.xyxyxyxy contains the 4 corner points (normalized or absolute)
            # obb_data.cls contains class IDs
            # obb_data.conf contains confidence scores

            if hasattr(obb_data, 'xyxyxyxy') and obb_data.xyxyxyxy is not None:
                boxes = obb_data.xyxyxyxy.cpu().numpy()  # Shape: (N, 4, 2)
                classes = obb_data.cls.cpu().numpy()
                confidences = obb_data.conf.cpu().numpy()

                for i in range(len(boxes)):
                    pred = obb_to_ellipse(
                        obb_points=boxes[i],
                        class_id=int(classes[i]),
                        confidence=float(confidences[i]),
                        image_path=rel_path
                    )
                    predictions.append(pred)

        return predictions

    def predict_directory(
        self,
        source_dir: str,
        recursive: bool = True
    ) -> List[EllipsePrediction]:
        """
        Run prediction on all images in a directory.

        Args:
            source_dir: Directory containing images
            recursive: Search subdirectories

        Returns:
            List of all predictions
        """
        all_predictions = []

        # Find all PNG images
        source_path = Path(source_dir)
        pattern = '**/*.png' if recursive else '*.png'

        image_files = list(source_path.glob(pattern))
        print(f"Found {len(image_files)} images in {source_dir}")

        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                print(f"Processing image {i + 1}/{len(image_files)}...")

            predictions = self.predict_image(str(img_path), source_dir)
            all_predictions.extend(predictions)

        print(f"Total predictions: {len(all_predictions)}")
        return all_predictions


def save_predictions_csv(
    predictions: List[EllipsePrediction],
    output_path: str,
    include_confidence: bool = False
):
    """
    Save predictions to CSV in competition format.

    CSV format:
        ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),
        ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification

    Args:
        predictions: List of EllipsePrediction objects
        output_path: Output CSV file path
        include_confidence: Include confidence column (not in competition format)
    """
    # Define columns
    columns = [
        'ellipseCenterX(px)',
        'ellipseCenterY(px)',
        'ellipseSemimajor(px)',
        'ellipseSemiminor(px)',
        'ellipseRotation(deg)',
        'inputImage',
        'crater_classification'
    ]

    if include_confidence:
        columns.append('confidence')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for pred in predictions:
            row = [
                f"{pred.center_x:.2f}",
                f"{pred.center_y:.2f}",
                f"{pred.semimajor:.2f}",
                f"{pred.semiminor:.2f}",
                f"{pred.rotation_deg:.2f}",
                pred.image_path,
                pred.classification
            ]

            if include_confidence:
                row.append(f"{pred.confidence:.4f}")

            writer.writerow(row)

    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict crater ellipses using YOLO OBB model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO OBB model (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV file path (default: predictions.csv)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=100,
        help='Maximum detections per image (default: 100)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device (e.g., 0 or cpu)'
    )
    parser.add_argument(
        '--include-confidence',
        action='store_true',
        help='Include confidence scores in output CSV'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = CraterPredictor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
        device=args.device
    )

    # Run predictions
    source_path = Path(args.source)

    if source_path.is_file():
        # Single image
        predictions = predictor.predict_image(
            str(source_path),
            str(source_path.parent)
        )
    else:
        # Directory
        predictions = predictor.predict_directory(
            str(source_path),
            recursive=not args.no_recursive
        )

    # Save results
    save_predictions_csv(
        predictions,
        args.output,
        include_confidence=args.include_confidence
    )


if __name__ == '__main__':
    main()
