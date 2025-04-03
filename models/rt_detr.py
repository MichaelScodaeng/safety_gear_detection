"""
Implementation of RT-DETR (Real-Time Detection Transformer) from Ultralytics for Safety Gear Detection.
"""

import os
import torch
from ultralytics import RTDETR

class RTDETR_Model:
    """
    Wrapper for Ultralytics RT-DETR (Real-Time Detection Transformer) model.
    RT-DETR combines CNN backbones with Transformer architectures for efficient object detection.
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize RT-DETR model

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config else {}
        self.model_type = config.get('model_type', 'rtdetr-l')
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the RT-DETR model"""
        # Map model type to actual model names in Ultralytics
        model_map = {
            'rtdetr-l': 'rtdetr-l.pt',
            'rtdetr-x': 'rtdetr-x.pt',
        }

        # Get the model name based on model_type
        model_name = model_map.get(self.model_type, 'rtdetr-l.pt')
        print(f"Initializing RT-DETR model: {model_name}")

        # Initialize the RTDETR model from Ultralytics
        try:
            self.model = RTDETR(model_name)
            print(f"RT-DETR model loaded successfully: {model_name}")
        except Exception as e:
            print(f"Error loading RT-DETR model: {e}")
            print("Falling back to creating a new RT-DETR model")
            self.model = RTDETR(model_name)

        # Move model to device
        self.model.to(self.device)

    def predict(self, images, conf=0.25, iou=0.45):
        """
        Run inference on images

        Args:
            images: Image(s) to run inference on (file path, numpy array, PIL image, or list)
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS

        Returns:
            list: List of Results objects from Ultralytics
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")

        # Run inference
        results = self.model.predict(
            source=images,
            conf=conf,
            iou=iou,
            device=self.device
        )

        return results

    def train(self, **kwargs):
        """
        Train the RT-DETR model

        Args:
            **kwargs: Training arguments to pass to the model

        Returns:
            Any: Training results from Ultralytics
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")

        # Default training parameters
        train_args = {
            'data': kwargs.get('data', 'data.yaml'),
            'epochs': kwargs.get('epochs', 100),
            'patience': kwargs.get('patience', 50),
            'batch': kwargs.get('batch', 16),
            'imgsz': kwargs.get('imgsz', 640),
            'device': self.device,
            'workers': kwargs.get('workers', 8),
            'project': kwargs.get('project', 'runs/train'),
            'name': kwargs.get('name', self.model_type),
            'exist_ok': kwargs.get('exist_ok', False),
            'pretrained': kwargs.get('pretrained', True),
            'optimizer': kwargs.get('optimizer', 'AdamW'),
            'lr0': kwargs.get('lr0', 0.001),
            'weight_decay': kwargs.get('weight_decay', 0.0001),
            'cos_lr': kwargs.get('cos_lr', True),
            'dropout': kwargs.get('dropout', 0.0),
            'label_smoothing': kwargs.get('label_smoothing', 0.0),
            'profile': kwargs.get('profile', False),
            'seed': kwargs.get('seed', 0),
        }

        # Update with any additional kwargs
        train_args.update({k: v for k, v in kwargs.items() if k not in train_args})

        # Train the model
        results = self.model.train(**train_args)

        return results

    def export(self, format='onnx', **kwargs):
        """
        Export the model to different formats

        Args:
            format (str): Export format (e.g., 'onnx', 'tflite', 'coreml', etc.)
            **kwargs: Additional export arguments

        Returns:
            str: Path to exported model
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")

        # Export the model
        export_path = self.model.export(format=format, **kwargs)

        return export_path

    def val(self, **kwargs):
        """
        Validate the model

        Args:
            **kwargs: Validation arguments

        Returns:
            Any: Validation results from Ultralytics
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")

        # Default validation parameters
        val_args = {
            'data': kwargs.get('data', 'data.yaml'),
            'batch': kwargs.get('batch', 16),
            'imgsz': kwargs.get('imgsz', 640),
            'device': self.device,
            'workers': kwargs.get('workers', 8),
            'project': kwargs.get('project', 'runs/val'),
            'name': kwargs.get('name', self.model_type),
            'exist_ok': kwargs.get('exist_ok', False),
        }

        # Update with any additional kwargs
        val_args.update({k: v for k, v in kwargs.items() if k not in val_args})

        # Validate the model
        results = self.model.val(**val_args)

        return results
