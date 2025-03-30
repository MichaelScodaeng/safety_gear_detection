"""
Visualization utilities for the Safety Gear Detection System.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def draw_boxes(image, boxes, labels, scores, class_names):
    """
    Draw bounding boxes on an image.

    Args:
        image (np.ndarray): Image to draw on
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format
        labels (np.ndarray): Class labels
        scores (np.ndarray): Confidence scores
        class_names (dict): Mapping from class IDs to class names

    Returns:
        np.ndarray: Image with bounding boxes
    """
    # Create a copy of the image
    image_copy = image.copy()

    # Generate random colors for each class
    colors = {}
    for label in set(labels):
        colors[label] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = colors[label]

        # Convert box coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        class_name = class_names.get(label, f"Class_{label}")
        label_text = f"{class_name}: {score:.2f}"

        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(
            image_copy,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image_copy,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image_copy


def show_image(image, figsize=(12, 12)):
    """
    Display an image using matplotlib.

    Args:
        image (np.ndarray): Image to display
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def plot_training_history(history, output_path=None):
    """
    Plot training history.

    Args:
        history (dict): Training history
        output_path (str, optional): Path to save the plot
    """
    if history.get('val_map') is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_map'], label='Val mAP')
        plt.title('mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        
        plt.show()


def visualize_prediction(detector, image, boxes=None, labels=None, scores=None, figsize=(12, 12)):
    """
    Visualize predictions on an image

    Args:
        detector: SafetyGearDetector instance
        image: Image to visualize (path, numpy array, or PIL Image)
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        labels: Class labels
        scores: Confidence scores
        figsize: Figure size for plotting

    Returns:
        np.ndarray: Image with bounding boxes
    """
    # Process image if needed
    if boxes is None or labels is None:
        # Run prediction if not provided
        boxes, labels, scores, _ = detector.predict(image)
    # Apply NMS to remove duplicate boxes if boxes were provided directly
    elif len(boxes) > 1:
        boxes, labels, scores = apply_nms(boxes, labels, scores, iou_threshold=0.3)

    # Convert image to numpy array if it's a path or PIL Image
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create a copy of the image
    image_with_boxes = image.copy()
    
    # Generate random colors for each class
    colors = {}
    class_names = detector.config.get('CLASS_NAMES', [])
    
    # Draw boxes
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get class color (consistent for same class)
        if label not in colors:
            colors[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        color = colors[label]
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Get class name
        class_name = class_names[label - 1] if label < len(class_names) + 1 else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            image_with_boxes,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_with_boxes,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Display the image
    plt.figure(figsize=figsize)
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return image_with_boxes


def visualize_debug_images(detector, images, targets, outputs=None, max_images=2):
    """
    Visualize sample images with bounding boxes for debugging purposes
    
    Args:
        detector: SafetyGearDetector instance
        images: List of image tensors
        targets: List of target dictionaries with ground truth boxes
        outputs: List of model prediction dictionaries (optional)
        max_images: Maximum number of images to visualize
    """
    num_images = min(len(images), max_images)
    
    for i in range(num_images):
        # Get the current image and target
        image = images[i]
        target = targets[i]
        
        # Convert image tensor to numpy array for visualization
        img_np = image.cpu().permute(1, 2, 0).numpy()
        
        # Normalize for display if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        elif img_np.max() > 255:
            img_np = (img_np / img_np.max() * 255).astype(np.uint8)
        
        # Create plot layout
        if outputs is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Debug Image {i+1}", fontsize=16)
            ax1.set_title("Ground Truth")
            ax2.set_title("Predictions")
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            fig.suptitle(f"Debug Image {i+1}", fontsize=16)
            ax1.set_title("Ground Truth")
        
        # Display image with ground truth boxes
        ax1.imshow(img_np)
        
        # Draw ground truth boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy() if 'labels' in target else [1] * len(boxes)
            
            for j, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x1, y1), width, height, 
                    linewidth=2, edgecolor='g', facecolor='none'
                )
                ax1.add_patch(rect)
                
                # Display label
                class_name = detector.config['CLASS_NAMES'][label-1] if label < len(detector.config['CLASS_NAMES'])+1 else f"Class {label}"
                ax1.text(
                    x1, y1-5, class_name, 
                    color='white', fontsize=8, 
                    bbox=dict(facecolor='g', alpha=0.8, pad=2)
                )
        
        # Display predictions if available
        if outputs is not None:
            output = outputs[i]
            ax2.imshow(img_np)
            
            # REPLACE THIS SECTION WITH YOUR CODE:
            if 'boxes' in output and len(output['boxes']) > 0:
                # Filter predictions by confidence
                confidence_threshold = 0.5
                if 'scores' in output:
                    mask = output['scores'] > confidence_threshold
                    boxes = output['boxes'][mask].cpu().numpy()
                    labels = output['labels'][mask].cpu().numpy()
                    scores = output['scores'][mask].cpu().numpy()
                else:
                    boxes = output['boxes'].cpu().numpy()
                    labels = output['labels'].cpu().numpy() if 'labels' in output else [1] * len(boxes)
                    scores = [1.0] * len(boxes)
                
                # Apply NMS to remove duplicate boxes
                if len(boxes) > 1:  # Only apply if we have multiple boxes
                    boxes, labels, scores = apply_nms(boxes, labels, scores, iou_threshold=0.3)
                
                # Now draw the filtered boxes
                for j, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create rectangle patch
                    rect = patches.Rectangle(
                        (x1, y1), width, height, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    # Display label with confidence
                    class_name = detector.config['CLASS_NAMES'][label-1] if label < len(detector.config['CLASS_NAMES'])+1 else f"Class {label}"
                    ax2.text(
                        x1, y1-5, f"{class_name}: {score:.2f}", 
                        color='white', fontsize=8, 
                        bbox=dict(facecolor='r', alpha=0.8, pad=2)
                    )
            plt.tight_layout()
            plt.show()
def apply_nms(boxes, labels, scores, iou_threshold=0.3):
    """Apply non-maximum suppression to remove overlapping boxes"""
    from torchvision.ops import nms
    import torch
    
    # Convert to tensors if they're numpy arrays
    if isinstance(boxes, np.ndarray):
        boxes_tensor = torch.from_numpy(boxes)
    else:
        boxes_tensor = boxes
        
    if isinstance(scores, np.ndarray):
        scores_tensor = torch.from_numpy(scores)
    else:
        scores_tensor = scores
    
    # Apply NMS for each class separately
    unique_labels = np.unique(labels)
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    for label in unique_labels:
        # Get indices for this class
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
            
        # Get boxes and scores for this class
        class_boxes = boxes_tensor[indices]
        class_scores = scores_tensor[indices]
        
        # Apply NMS
        keep_indices = nms(class_boxes, class_scores, iou_threshold)
        
        # Keep the selected boxes
        for idx in keep_indices:
            original_idx = indices[idx]
            keep_boxes.append(boxes[original_idx])
            keep_scores.append(scores[original_idx])
            keep_labels.append(labels[original_idx])
    
    if keep_boxes:
        return np.array(keep_boxes), np.array(keep_labels), np.array(keep_scores)
    else:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

def visualize_yolo_results(results, image=None, figsize=(12, 12), save_path=None):
    """
    Visualize results from a YOLO model prediction or training
    
    Args:
        results: Results from YOLO model.predict() or model.train()
        image: Original image (optional, if not provided will use the image from results)
        figsize: Figure size for the plot
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure with visualized results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if results is from training (DetMetrics) or prediction (Results)
    if hasattr(results, 'box') and hasattr(results, 'metrics'):
        # This is training metrics, plot training graphs
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("YOLO Training Results", fontsize=16)
        
        # Extract metrics
        metrics = results.metrics
        
        # Plot precision-recall curve if available
        if hasattr(results, 'pr_curve') and results.pr_curve is not None:
            for i, class_name in enumerate(results.names):
                if i < len(results.pr_curve):
                    precision = results.pr_curve[i][:, 0]
                    recall = results.pr_curve[i][:, 1]
                    ax[0].plot(recall, precision, label=class_name)
            
            ax[0].set_xlabel('Recall')
            ax[0].set_ylabel('Precision')
            ax[0].set_title('Precision-Recall Curve')
            ax[0].legend()
        
        # Plot mAP progress if available
        if hasattr(results, 'maps') and results.maps is not None:
            epochs = range(1, len(results.maps) + 1)
            ax[1].plot(epochs, results.maps)
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('mAP50-95')
            ax[1].set_title('mAP Progress')
        
        # Plot loss progress if available
        if hasattr(results, 'box_loss') and results.box_loss is not None:
            epochs = range(1, len(results.box_loss) + 1)
            ax[2].plot(epochs, results.box_loss, label='Box Loss')
            if hasattr(results, 'cls_loss') and results.cls_loss is not None:
                ax[2].plot(epochs, results.cls_loss, label='Class Loss')
            if hasattr(results, 'dfl_loss') and results.dfl_loss is not None:
                ax[2].plot(epochs, results.dfl_loss, label='DFL Loss')
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Loss')
            ax[2].set_title('Training Loss')
            ax[2].legend()
        
        # Print metrics summary
        print(f"Final mAP50-95: {results.map50_95:.3f}")
        print(f"Final mAP50: {results.map50:.3f}")
        
    else:
        # This is prediction results, visualize detections
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use image from results if not provided
        if image is None:
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'orig_img'):
                image = results[0].orig_img
            else:
                raise ValueError("Image not provided and not found in results")
        
        # Display image
        if isinstance(image, str):
            # Load image if it's a path
            image = plt.imread(image)
        
        ax.imshow(image)
        
        # Process results based on type
        if isinstance(results, list) and len(results) > 0:
            # For prediction results
            if hasattr(results[0], 'names'):
                num_classes = len(results[0].names)
                colors = plt.cm.hsv(np.linspace(0, 1, num_classes))
                
                # Draw detections
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for box, cl, conf in zip(boxes, cls, confs):
                        x1, y1, x2, y2 = box
                        color = colors[cl]
                        
                        # Draw rectangle
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor=color, linewidth=2)
                        ax.add_patch(rect)
                        
                        # Add label
                        class_name = r.names[cl] if hasattr(r, 'names') else f"Class {cl}"
                        ax.text(x1, y1-5, f"{class_name} {conf:.2f}", 
                               color='white', fontsize=10,
                               bbox=dict(facecolor=color, alpha=0.7))
                
                ax.set_title(f"YOLO Detection Results")
                ax.axis('off')
            else:
                ax.set_title("No detection results found")
        else:
            ax.set_title("No valid results provided")
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
    
    return fig