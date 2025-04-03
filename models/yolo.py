"""
Implementation of YOLO models using Ultralytics for Safety Gear Detection.
"""

import os
import yaml
import torch
from ultralytics import YOLO

class UltralyticsYOLO:
    """Wrapper for Ultralytics YOLO models (v4, v8, v12)"""
    
    def __init__(self, num_classes=80, model_type='yolov8n', device='cuda', config=None):
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device
        self.config = config or {}
        
        # Extract YOLO version and size
        if model_type.startswith('yolov8'):
            # For YOLOv8 models
            self.model = YOLO(f'{model_type}.pt')
        elif model_type.startswith('yolov4'):
            import os
            from urllib.request import urlretrieve
            
            model_path = f'{model_type}.pt'
            if not os.path.exists(model_path):
                print(f"Downloading {model_type} weights...")
                url = f"https://github.com/ultralytics/yolov4/releases/download/v1.0/{model_type}.pt"
                urlretrieve(url, model_path)
                print(f"Downloaded {model_type} weights to {model_path}")
            
            self.model = YOLO(model_path)
        elif model_type.startswith('yolo12'):
            # For YOLOv12 models
            self.model = YOLO(f'{model_type}.pt')
        else:
            raise ValueError(f"Unsupported YOLO model type: {model_type}")
    
    def _initialize_model(self):
        """Initialize the YOLO model using Ultralytics"""
        # Parse model type and size
        if self.model_type.startswith(('yolov4', 'yolov8', 'yolov12')):
            # Extract version and size (e.g., 'yolov8n' -> '8', 'n')
            if len(self.model_type) >= 6:
                version = self.model_type[4:6].strip('v')
                size = self.model_type[6:] if len(self.model_type) > 6 else 'n'
            else:
                version = self.model_type[4:].strip('v')
                size = 'n'  # Default to nano size
                
            # Construct model path
            model_path = f'yolov{version}{size}.pt'
            
            # Check if a custom weights file is specified
            if 'weights' in self.config and self.config['weights']:
                if os.path.exists(self.config['weights']):
                    model_path = self.config['weights']
            
            # Load model with Ultralytics
            try:
                self.model = YOLO(model_path)
                print(f"Loaded {self.model_type} model: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load {model_path}: {e}")
            
            # Set up training configuration
            self.set_up_training_config()
        else:
            raise ValueError(f"Unsupported YOLO model type: {self.model_type}")
    
    def set_up_training_config(self):
        """Set up training configuration for the YOLO model"""
        # Create data.yaml file if not provided
        if 'data_yaml' not in self.config or not self.config['data_yaml']:
            # Create a temporary YAML file
            data_yaml = {
                'path': os.path.dirname(self.config.get('train_dir', '.')),
                'train': os.path.join('train', 'images'),
                'val': os.path.join('valid', 'images'),
                'test': os.path.join('test', 'images'),
                'names': {i: name for i, name in enumerate(self.config.get('class_names', [f'class_{i}' for i in range(self.num_classes)]))}
            }
            
            # Save to disk
            yaml_path = os.path.join(self.config.get('output_path', '.'), 'data.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            
            self.config['data_yaml'] = yaml_path
            print(f"Created data.yaml at {yaml_path}")
    
    def train(self, data=None, train_data=None, val_data=None, epochs=100, batch_size=16, img_size=640, **kwargs):
        """
        Train the YOLO model
        
        Args:
            data (str): Path to data.yaml file 
            train_data: Training data (not used directly, Ultralytics uses data.yaml)
            val_data: Validation data (not used directly, Ultralytics uses data.yaml)
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
            img_size (int): Image size for training
            **kwargs: Additional parameters to pass to the YOLO model
        
        Returns:
            results: Training results from Ultralytics
        """
        # Training parameters
        params = {
            'data': data if data else self.config.get('data_yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'project': self.config.get('project', 'safety_gear_detection'),
            'name': self.config.get('run_name', self.model_type),
        }
        
        # Add any additional parameters from kwargs
        params.update(kwargs)
        
        # Additional training parameters from config
        if 'lr0' in self.config:
            params['lr0'] = self.config['lr0']
        if 'weight_decay' in self.config:
            params['weight_decay'] = self.config['weight_decay']
        
        # Start training
        print(f"Starting {self.model_type} training with parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        results = self.model.train(**params)
        return results
    
    def evaluate(self, data=None):
        """
        Evaluate the model
        
        Args:
            data: Validation data (not used directly, uses data.yaml)
            
        Returns:
            results: Evaluation results
        """
        # Evaluation parameters
        params = {
            'data': self.config.get('data_yaml'),
            'imgsz': self.config.get('img_size', 640),
            'batch': self.config.get('batch_size', 16),
            'device': self.device,
        }
        
        # Start evaluation
        results = self.model.val(**params)
        return results
    
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run prediction on an image
        
        Args:
            image: Image to predict on (path, numpy array, or tensor)
            conf_threshold (float): Confidence threshold
            iou_threshold (float): NMS threshold
            
        Returns:
            results: Prediction results from Ultralytics
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        return results
    
    
    def export(self, format='onnx'):
        """
        Export the model to different formats
        
        Args:
            format (str): Format to export to ('onnx', 'torchscript', etc.)
            
        Returns:
            path: Path to the exported model
        """
        path = self.model.export(format=format)
        return path
    
    def print_hyperparameters(self, training_args=None):
        """
        Print detailed information about model hyperparameters and training settings
        
        Args:
            training_args (dict, optional): Training arguments including learning rate, 
                                        batch size, epochs, etc.
        """
        try:
            from tabulate import tabulate
            use_tabulate = True
        except ImportError:
            use_tabulate = False
            
        print("\n" + "="*80)
        print(f"{self.model_type.upper()} MODEL CONFIGURATION")
        print("="*80)
        
        # Model information
        model_info = [
            ["Model Type", self.model_type],
            ["Number of Classes", self.num_classes],
            ["Device", self.device],
        ]
        
        # Get model details from Ultralytics
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
            yaml_cfg = self.model.model.yaml
            for key, value in yaml_cfg.items():
                if key not in ['backbone', 'head', 'nc'] and not isinstance(value, dict) and not isinstance(value, list):
                    model_info.append([key, value])
        
        # Print model information
        if use_tabulate:
            print(tabulate(model_info, headers=["Parameter", "Value"], tablefmt="grid"))
        else:
            for param, value in model_info:
                print(f"{param}: {value}")
        
        # Print training parameters if provided
        if training_args:
            print("\n" + "="*80)
            print("TRAINING CONFIGURATION")
            print("="*80)
            
            training_info = []
            for key, value in training_args.items():
                training_info.append([key, value])
            
            if use_tabulate:
                print(tabulate(training_info, headers=["Parameter", "Value"], tablefmt="grid"))
            else:
                for param, value in training_info:
                    print(f"{param}: {value}")
        
        print("="*80)
        print("End of Configuration")
        print("="*80)
    