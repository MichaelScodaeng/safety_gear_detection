"""
Main detector class for the Safety Gear Detection System.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import CFG
from models.rcnn import RCNN
from models.fast_rcnn import FastRCNN
from models.faster_rcnn import FasterRCNN_Model


class SafetyGearDetector:
    """
    Class to detect safety gear using R-CNN, Fast R-CNN, or Faster R-CNN models
    """

    def __init__(self, model_type='faster_rcnn', device=None, config=None):
        """
        Initialize the detector

        Args:
            model_type (str): Type of model to use ('rcnn', 'fast_rcnn', or 'faster_rcnn')
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        self.model_type = model_type.lower()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use default config if none provided
        if config is None:
            self.config = {
                'class_names': CFG.PPE_CLASSES,
                'output_path': CFG.MODEL_PATH
            }
        else:
            self.config = config

        # Number of classes
        self.num_classes = len(self.config['class_names'])

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on model type"""
        if self.model_type == 'rcnn':
            self.model = RCNN(self.num_classes, self.device, self.config)
        elif self.model_type == 'fast_rcnn':
            self.model = FastRCNN(self.num_classes, self.device, self.config)
        elif self.model_type == 'faster_rcnn':
            self.model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, train_loader, valid_loader=None, epochs=10, 
              lr=0.001, weight_decay=0.0005, batch_size=4, 
              fine_tune=False, freeze_backbone=True):
        """
        Train the model

        Args:
            train_loader (DataLoader): DataLoader for training data
            valid_loader (DataLoader, optional): DataLoader for validation data
            epochs (int): Number of epochs to train
            lr (float): Learning rate
            weight_decay (float): Weight decay for optimizer
            batch_size (int): Batch size for training
            fine_tune (bool): Whether to fine-tune an existing model
            freeze_backbone (bool): Whether to freeze the backbone during fine-tuning

        Returns:
            dict: Training history
        """
        # Fine-tune if requested
        if fine_tune and self.model_type == 'faster_rcnn':
            self.model.fine_tune(freeze_backbone=freeze_backbone)

        # Train the model
        history = self.model.train(
            train_loader,
            valid_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size
        )

        return history

    def predict(self, image, confidence_threshold=0.5, nms_threshold=0.3):
        """
        Run inference on an image

        Args:
            image: Image to run inference on (path, numpy array, PIL image, or tensor)
            confidence_threshold (float): Confidence threshold for detections
            nms_threshold (float): NMS threshold

        Returns:
            tuple: (boxes, labels, scores, class_names)
        """
        # Check if image is a path
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Image not found: {image}")

        # Run prediction
        boxes, labels, scores = self.model.predict(
            image,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        # Get class names
        class_names = [self.config['class_names'].get(label, f"Class_{label}") for label in labels]

        return boxes, labels, scores, class_names

    def visualize_prediction(self, image, boxes=None, labels=None, scores=None, figsize=(12, 12)):
        """
        Visualize predictions on an image

        Args:
            image: Image to visualize (path, numpy array, PIL image, or tensor)
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            labels: Class labels
            scores: Confidence scores
            figsize (tuple): Figure size for matplotlib

        Returns:
            np.ndarray: Image with visualizations
        """
        # Check if image is a path
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Image not found: {image}")

        # If no predictions provided, run prediction
        if boxes is None or labels is None or scores is None:
            boxes, labels, scores, _ = self.predict(image)

        # Create a copy of the image
        image_copy = image.copy()

        # Generate random colors for each class
        colors = {}
        for label in set(labels):
            colors[label] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
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
            class_name = self.config['class_names'].get(label, f"Class_{label}")
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

        # Display the image
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return image_copy

    def save_model(self, filepath=None):
        """
        Save the model to disk

        Args:
            filepath (str, optional): Path to save the model
        """
        if filepath is None:
            # Create default path
            os.makedirs(self.config['output_path'], exist_ok=True)
            filepath = f"{self.config['output_path']}/{self.model_type}_safety_gear.pt"

        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a model from disk

        Args:
            filepath (str): Path to the saved model
        """
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def evaluate(self, data_loader, iou_threshold=0.5):
        """
        Evaluate the model on a dataset

        Args:
            data_loader (DataLoader): DataLoader for evaluation
            iou_threshold (float): IoU threshold for mAP calculation

        Returns:
            dict: Evaluation metrics
        """
        return self.model.evaluate(data_loader, iou_threshold)