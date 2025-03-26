"""
Visualization utilities for the Safety Gear Detection System.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


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
