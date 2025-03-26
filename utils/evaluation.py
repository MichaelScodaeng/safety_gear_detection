"""
Evaluation utilities for the Safety Gear Detection System.
"""

import numpy as np
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.

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


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision).

    Args:
        predictions (list): List of dictionaries with 'boxes', 'labels', and 'scores'
        targets (list): List of dictionaries with 'boxes' and 'labels'
        iou_threshold (float): IoU threshold for considering a detection as correct

    Returns:
        float: mAP value
    """
    # Convert predictions and targets to format expected by _calculate_map
    all_pred_boxes = [p['boxes'] for p in predictions]
    all_pred_scores = [p['scores'] for p in predictions]
    all_pred_labels = [p['labels'] for p in predictions]
    all_gt_boxes = [t['boxes'] for t in targets]
    all_gt_labels = [t['labels'] for t in targets]

    return _calculate_map(all_pred_boxes, all_pred_scores, all_pred_labels,
                         all_gt_boxes, all_gt_labels, iou_threshold)


def _calculate_map(all_pred_boxes, all_pred_scores, all_pred_labels,
                  all_gt_boxes, all_gt_labels, iou_threshold=0.5, num_classes=10):
    """
    Calculate mean Average Precision.

    Args:
        all_pred_boxes (list): List of predicted boxes for each image
        all_pred_scores (list): List of prediction scores for each image
        all_pred_labels (list): List of predicted labels for each image
        all_gt_boxes (list): List of ground truth boxes for each image
        all_gt_labels (list): List of ground truth labels for each image
        iou_threshold (float): IoU threshold for mAP calculation
        num_classes (int): Number of classes

    Returns:
        float: mAP value
    """
    aps = []

    # Process each class
    for c in range(num_classes):
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
                    iou = calculate_iou(pred_box, gt_box)
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
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
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


def plot_confusion_matrix(predictions, targets, class_names, output_path=None):
    """
    Plot confusion matrix for object detection results.

    Args:
        predictions (list): List of dictionaries with 'boxes', 'labels', and 'scores'
        targets (list): List of dictionaries with 'boxes' and 'labels'
        class_names (dict): Dictionary mapping class IDs to class names
        output_path (str, optional): Path to save the confusion matrix plot
    """
    # Extract predicted and ground truth labels
    all_pred_labels = []
    all_gt_labels = []

    for i in range(len(predictions)):
        # Match predictions to ground truth based on IoU
        pred_boxes = predictions[i]['boxes']
        pred_labels = predictions[i]['labels']
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']

        # Skip if no predictions or ground truths
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        # Convert boxes to torch tensors for IoU calculation
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes)

        # Calculate IoU between all pairs of boxes
        iou_matrix = box_iou(pred_boxes, gt_boxes)

        # Match predictions to ground truth
        matched_gt_indices = torch.argmax(iou_matrix, dim=1)
        valid_matches = torch.max(iou_matrix, dim=1)[0] >= 0.5

        for j in range(len(pred_labels)):
            if valid_matches[j]:
                # Matched prediction
                all_pred_labels.append(pred_labels[j])
                all_gt_labels.append(gt_labels[matched_gt_indices[j]])
            else:
                # False positive (no match)
                all_pred_labels.append(pred_labels[j])
                # Use -1 to represent no ground truth match
                all_gt_labels.append(-1)

        # Add unmatched ground truth (false negatives)
        matched_gt_mask = torch.zeros(len(gt_labels), dtype=torch.bool)
        for j in range(len(pred_labels)):
            if valid_matches[j]:
                matched_gt_mask[matched_gt_indices[j]] = True

        for j in range(len(gt_labels)):
            if not matched_gt_mask[j]:
                # Add false negative
                # Use -1 to represent no prediction match
                all_pred_labels.append(-1)
                all_gt_labels.append(gt_labels[j])

    # Convert to numpy arrays
    all_pred_labels = np.array(all_pred_labels)
    all_gt_labels = np.array(all_gt_labels)

    # Define class labels for confusion matrix
    class_ids = sorted(class_names.keys())
    class_names_list = [class_names.get(i, f"Class_{i}") for i in class_ids]
    
    # Add background class for unmatched detections
    cm_labels = ['Background'] + class_names_list
    
    # Adjust label values to include background
    all_pred_labels_adj = np.array([l + 1 if l >= 0 else 0 for l in all_pred_labels])
    all_gt_labels_adj = np.array([l + 1 if l >= 0 else 0 for l in all_gt_labels])

    # Calculate confusion matrix
    cm = confusion_matrix(all_gt_labels_adj, all_pred_labels_adj, 
                         labels=range(len(cm_labels)))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()


def plot_precision_recall_curve(precisions, recalls, ap, class_name, output_path=None):
    """
    Plot precision-recall curve for a class.

    Args:
        precisions (list): List of precision values
        recalls (list): List of recall values
        ap (float): Average precision value
        class_name (str): Class name
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {class_name}\nAP = {ap:.4f}')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()


def analyze_detections(predictions, targets, class_names, confidence_threshold=0.5):
    """
    Analyze detection results and provide statistics.

    Args:
        predictions (list): List of dictionaries with 'boxes', 'labels', and 'scores'
        targets (list): List of dictionaries with 'boxes' and 'labels'
        class_names (dict): Dictionary mapping class IDs to class names
        confidence_threshold (float): Confidence threshold for detections

    Returns:
        dict: Dictionary containing detection statistics
    """
    # Initialize counters
    stats = {
        'total_predictions': 0,
        'total_ground_truth': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'class_stats': {},
        'overall_precision': 0,
        'overall_recall': 0,
        'overall_f1': 0
    }
    
    # Initialize per-class counters
    for class_id, class_name in class_names.items():
        stats['class_stats'][class_id] = {
            'name': class_name,
            'predictions': 0,
            'ground_truth': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'ap': 0
        }
    
    # Process each image
    for i in range(len(predictions)):
        # Get predictions and ground truth
        pred_boxes = predictions[i]['boxes']
        pred_labels = predictions[i]['labels']
        pred_scores = predictions[i]['scores']
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']
        
        # Filter by confidence threshold
        keep = pred_scores >= confidence_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        # Update total counts
        stats['total_predictions'] += len(pred_boxes)
        stats['total_ground_truth'] += len(gt_boxes)
        
        # Update per-class counts
        for label in pred_labels:
            if label in stats['class_stats']:
                stats['class_stats'][label]['predictions'] += 1
        
        for label in gt_labels:
            if label in stats['class_stats']:
                stats['class_stats'][label]['ground_truth'] += 1
        
        # Match predictions to ground truth
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Convert boxes to torch tensors for IoU calculation
            if not isinstance(pred_boxes, torch.Tensor):
                pred_boxes = torch.tensor(pred_boxes)
            if not isinstance(gt_boxes, torch.Tensor):
                gt_boxes = torch.tensor(gt_boxes)
            
            # Calculate IoU between all pairs of boxes
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            
            # Match predictions to ground truth
            matched_gt_indices = torch.argmax(iou_matrix, dim=1)
            valid_matches = torch.max(iou_matrix, dim=1)[0] >= 0.5
            
            # Mark matched ground truth
            gt_matched = torch.zeros(len(gt_labels), dtype=torch.bool)
            
            # Process each prediction
            for j in range(len(pred_labels)):
                if valid_matches[j]:
                    # True positive
                    gt_idx = matched_gt_indices[j].item()
                    pred_label = pred_labels[j]
                    gt_label = gt_labels[gt_idx]
                    
                    if pred_label == gt_label:
                        stats['true_positives'] += 1
                        if pred_label in stats['class_stats']:
                            stats['class_stats'][pred_label]['true_positives'] += 1
                        
                        # Mark ground truth as matched
                        gt_matched[gt_idx] = True
                    else:
                        # Wrong class prediction
                        stats['false_positives'] += 1
                        if pred_label in stats['class_stats']:
                            stats['class_stats'][pred_label]['false_positives'] += 1
                else:
                    # False positive (no match)
                    stats['false_positives'] += 1
                    if pred_labels[j] in stats['class_stats']:
                        stats['class_stats'][pred_labels[j]]['false_positives'] += 1
            
            # Count false negatives (unmatched ground truth)
            for j in range(len(gt_labels)):
                if not gt_matched[j]:
                    stats['false_negatives'] += 1
                    if gt_labels[j] in stats['class_stats']:
                        stats['class_stats'][gt_labels[j]]['false_negatives'] += 1
    
    # Calculate precision, recall, and F1 score for each class
    for class_id in stats['class_stats']:
        class_stat = stats['class_stats'][class_id]
        
        tp = class_stat['true_positives']
        fp = class_stat['false_positives']
        fn = class_stat['false_negatives']
        
        # Calculate precision
        class_stat['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate recall
        class_stat['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        precision = class_stat['precision']
        recall = class_stat['recall']
        class_stat['f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate overall precision, recall, and F1 score
    tp = stats['true_positives']
    fp = stats['false_positives']
    fn = stats['false_negatives']
    
    stats['overall_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    stats['overall_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision = stats['overall_precision']
    recall = stats['overall_recall']
    stats['overall_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return stats
                