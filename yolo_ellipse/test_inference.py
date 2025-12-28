"""
Inference Module for YOLO OBB Crater Detection

This module provides functions to:
1. Load a trained YOLO OBB model
2. Run inference on test images
3. Convert OBB predictions to ellipse parameters
4. Save results to CSV in competition format
"""

import os
import csv
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

# Image dimensions
IMG_WIDTH = 2592
IMG_HEIGHT = 2048


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


def obb_to_ellipse(obb_points: np.ndarray, class_id: int, confidence: float, image_path: str) -> EllipsePrediction:
    """
    Convert OBB (4 corner points) to ellipse parameters.
    
    For inscribed ellipse:
    - Center = OBB center
    - Semimajor = half of longer edge
    - Semiminor = half of shorter edge
    - Rotation = angle of longer edge
    """
    points = np.array(obb_points).reshape(4, 2)
    
    # Calculate center
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    # Calculate edge vectors
    edge1 = points[1] - points[0]
    edge2 = points[3] - points[0]
    
    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)
    
    # Determine semimajor (longer) and semiminor (shorter)
    if len1 >= len2:
        semimajor = len1 / 2
        semiminor = len2 / 2
        rotation_rad = math.atan2(edge1[1], edge1[0])
    else:
        semimajor = len2 / 2
        semiminor = len1 / 2
        rotation_rad = math.atan2(edge2[1], edge2[0])
    
    # Convert to degrees and normalize to 0-360
    rotation_deg = math.degrees(rotation_rad) % 360
    if rotation_deg < 0:
        rotation_deg += 360
    
    return EllipsePrediction(
        center_x=center_x,
        center_y=center_y,
        semimajor=semimajor,
        semiminor=semiminor,
        rotation_deg=rotation_deg,
        classification=class_id,
        confidence=confidence,
        image_path=image_path
    )


def get_relative_path(image_path: str, base_dir: str) -> str:
    """Convert absolute path to relative path format for CSV."""
    rel_path = os.path.relpath(image_path, base_dir)
    
    # Remove .png extension
    if rel_path.endswith('.png'):
        rel_path = rel_path[:-4]
    
    # Normalize separators
    rel_path = rel_path.replace('\\', '/')
    
    # Handle flattened filenames
    stem = Path(rel_path).stem
    if '_' in stem and '/' not in rel_path:
        parts = stem.split('_')
        if len(parts) >= 4 and parts[0].startswith('altitude'):
            altitude, longitude = parts[0], parts[1]
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
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to trained YOLO OBB model (.pt file)
            conf_threshold: Confidence threshold (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
            max_det: Max detections per image (default: 100)
            device: Device ('', '0', 'cpu')
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device
        print("Model loaded successfully!")
    
    def predict_image(self, image_path: str, base_dir: str = '') -> List[EllipsePrediction]:
        """Run prediction on a single image."""
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device if self.device else None,
            verbose=False
        )
        
        predictions = []
        rel_path = get_relative_path(image_path, base_dir) if base_dir else image_path
        
        for result in results:
            if result.obb is None:
                continue
            
            obb_data = result.obb
            if hasattr(obb_data, 'xyxyxyxy') and obb_data.xyxyxyxy is not None:
                boxes = obb_data.xyxyxyxy.cpu().numpy()
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
    
    def predict_directory(self, source_dir: str, recursive: bool = True) -> List[EllipsePrediction]:
        """Run prediction on all images in a directory."""
        all_predictions = []
        source_path = Path(source_dir)
        pattern = '**/*.png' if recursive else '*.png'
        
        image_files = list(source_path.glob(pattern))
        # Filter out truth folder images
        image_files = [f for f in image_files if 'truth' not in str(f)]
        
        print(f"Found {len(image_files)} images in {source_dir}")
        
        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{len(image_files)}...")
            
            predictions = self.predict_image(str(img_path), source_dir)
            all_predictions.extend(predictions)
        
        print(f"Total predictions: {len(all_predictions)}")
        return all_predictions


def save_predictions_to_csv(
    predictions: List[EllipsePrediction],
    output_path: str,
    include_confidence: bool = False
):
    """
    Save predictions to CSV in competition format.
    
    Format: ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),
            ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
    """
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


def run_inference(
    model_path: str,
    test_dir: str,
    output_csv: str = 'predictions.csv',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 100,
    device: str = '',
    include_confidence: bool = False
) -> List[EllipsePrediction]:
    """
    Main inference function - run predictions and save to CSV.
    
    Args:
        model_path: Path to trained YOLO OBB model (.pt)
        test_dir: Directory containing test images
        output_csv: Output CSV file path
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Max detections per image
        device: Device to use
        include_confidence: Include confidence in output
    
    Returns:
        List of EllipsePrediction objects
    """
    # Initialize predictor
    predictor = CraterPredictor(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        device=device
    )
    
    # Run predictions
    predictions = predictor.predict_directory(test_dir)
    
    # Save to CSV
    save_predictions_to_csv(predictions, output_csv, include_confidence)
    
    return predictions



# Visualization utility functions

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms




def visualize_predictions_on_image(
    image_path: str,
    predictions: List[EllipsePrediction],
    save_path: str = None,
    show: bool = True
):
    """
    Visualize ellipse predictions on an image.
    
    Args:
        image_path: Path to the image
        predictions: List of EllipsePrediction for this image
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.imshow(img)
    
    # Color map for classes
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for pred in predictions:
        # Create ellipse patch
        ellipse = Ellipse(
            (pred.center_x, pred.center_y),
            width=2 * pred.semimajor,
            height=2 * pred.semiminor,
            angle=pred.rotation_deg,
            fill=False,
            edgecolor=colors[pred.classification % len(colors)],
            linewidth=2
        )
        ax.add_patch(ellipse)
        
        # Add label
        ax.text(
            pred.center_x, pred.center_y - pred.semimajor - 10,
            f"C{pred.classification} ({pred.confidence:.2f})",
            fontsize=8,
            color=colors[pred.classification % len(colors)],
            ha='center'
        )
    
    ax.set_title(f"Predictions: {len(predictions)} craters detected")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_sample_predictions(
    predictor: CraterPredictor,
    image_dir: str,
    num_samples: int = 3,
    save_dir: str = None
):
    """
    Visualize predictions on random sample images.
    
    Args:
        predictor: CraterPredictor instance
        image_dir: Directory containing images
        num_samples: Number of sample images to visualize
        save_dir: Optional directory to save visualizations
    """
    import random
    
    image_files = list(Path(image_dir).glob('**/*.png'))
    image_files = [f for f in image_files if 'truth' not in str(f)]
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for i, img_path in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}: {img_path}")
        
        # Get predictions for this image
        preds = predictor.predict_image(str(img_path), image_dir)
        
        save_path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = f"{save_dir}/sample_{i+1}_{img_path.stem}.png"
        
        visualize_predictions_on_image(str(img_path), preds, save_path=save_path)
