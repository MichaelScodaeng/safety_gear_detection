"""
Base RCNN implementation for the Safety Gear Detection System.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from data.dataset import get_transforms

class RCNNBase:
    """Base class for R-CNN family implementations"""

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize the base R-CNN class

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config else {}
        self.model = None
        self.class_names = self.config.get('class_names', {i: f"Class_{i}" for i in range(num_classes)})

    def _get_transform(self, train=False):
        """
        Get transforms for data preprocessing

        Args:
            train (bool): Whether to use training transforms

        Returns:
            A.Compose: Composition of transforms
        """
        return get_transforms(train)

    def save_model(self, filepath):
        """
        Save the model to disk

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_type': self.__class__.__name__,
            'config': self.config,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a model from disk

        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Verify model type
        if checkpoint.get('model_type') != self.__class__.__name__:
            print(f"Warning: Loading {checkpoint.get('model_type')} model into {self.__class__.__name__} class")

        # Update attributes
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.class_names = checkpoint.get('class_names', self.class_names)

        # Initialize the model before loading state dict
        self._initialize_model()

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {filepath}")

    def _initialize_model(self):
        """Initialize the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _initialize_model")

    def _collate_fn(self, batch):
        """
        Custom collate function for data loader

        Args:
            batch: Batch of (image, target) tuples

        Returns:
            tuple: (images, targets)
        """
        images = []
        targets = []

        for img, target in batch:
            images.append(img)
            targets.append(target)

        return images, targets

    def evaluate(self, data_loader, iou_threshold=0.5):
        """
        Evaluate the model on a dataset

        Args:
            data_loader (DataLoader): DataLoader for evaluation
            iou_threshold (float): IoU threshold for mAP calculation

        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()

        all_boxes = []
        all_scores = []
        all_labels = []
        all_gt_boxes = []
        all_gt_labels = []

        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]

                # Get model predictions
                outputs = self.model(images)

                for i, output in enumerate(outputs):
                    # Get predictions
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()

                    # Get ground truth
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    gt_labels = targets[i]['labels'].cpu().numpy()

                    all_boxes.append(pred_boxes)
                    all_scores.append(pred_scores)
                    all_labels.append(pred_labels)
                    all_gt_boxes.append(gt_boxes)
                    all_gt_labels.append(gt_labels)

        # Calculate mAP
        mAP = self._calculate_map(all_boxes, all_scores, all_labels,
                                  all_gt_boxes, all_gt_labels, iou_threshold)

        return {'mAP': mAP}

    def _calculate_map(self, all_pred_boxes, all_pred_scores, all_pred_labels,
                       all_gt_boxes, all_gt_labels, iou_threshold=0.5):
        """
        Calculate mean Average Precision

        Args:
            all_pred_boxes (list): List of predicted boxes for each image
            all_pred_scores (list): List of prediction scores for each image
            all_pred_labels (list): List of predicted labels for each image
            all_gt_boxes (list): List of ground truth boxes for each image
            all_gt_labels (list): List of ground truth labels for each image
            iou_threshold (float): IoU threshold for mAP calculation

        Returns:
            float: mAP value
        """
        aps = []

        # Process each class
        for c in range(1, self.num_classes + 1):  # Start from 1 to skip background class
            # Collect all predictions and ground truths for this class
            preds = []
            gts = []

            for i in range(len(all_pred_boxes)):
                # Get predictions for this class
                idx = all_pred_labels[i] == c
                boxes = all_pred_boxes[i][idx]
                scores = all_pred_scores[i][idx]

                # Add to predictions list with image index
                for box, score in zip(boxes, scores):
                    preds.append((i, box, score))

                # Get ground truths for this class
                gt_idx = all_gt_labels[i] == c
                gt_boxes = all_gt_boxes[i][gt_idx]

                # Add to ground truths list with image index and matched flag
                for gt_box in gt_boxes:
                    gts.append((i, gt_box, False))

            # Sort predictions by confidence score (descending)
            preds.sort(key=lambda x: x[2], reverse=True)

            # Calculate precision and recall
            tp = 0
            fp = 0
            precisions = []
            recalls = []

            for i, (img_idx, pred_box, _) in enumerate(preds):
                # Get ground truths for this image
                img_gts = [gt for gt in gts if gt[0] == img_idx and not gt[2]]

                if not img_gts:
                    # No ground truths for this image, false positive
                    fp += 1
                else:
                    # Calculate IoU with all ground truths
                    ious = []
                    for j, (_, gt_box, _) in enumerate(img_gts):
                        iou = self._calculate_iou(pred_box, gt_box)
                        ious.append((j, iou))

                    # Get the ground truth with highest IoU
                    best_match = max(ious, key=lambda x: x[1]) if ious else (None, 0)

                    if best_match[1] >= iou_threshold:
                        # Mark the ground truth as matched
                        img_gts[best_match[0]] = (img_gts[best_match[0]][0],
                                                 img_gts[best_match[0]][1], True)
                        tp += 1
                    else:
                        fp += 1

                # Calculate precision and recall
                precision = tp / (tp + fp)
                recall = tp / len(gts) if gts else 0

                precisions.append(precision)
                recalls.append(recall)

            # Calculate AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if not recalls:
                    continue

                # Get precision at recall greater than t
                prec = [p for p, r in zip(precisions, recalls) if r >= t]
                if prec:
                    ap += max(prec) / 11

            aps.append(ap)

        return np.mean(aps) if aps else 0

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes

        Args:
            box1: First box in format [x1, y1, x2, y2]
            box2: Second box in format [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate area of intersection
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h

        # Calculate area of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate IoU
        union = box1_area + box2_area - intersection
        iou = intersection / union if union > 0 else 0

        return iou