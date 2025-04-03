"""
Main detector class for the Safety Gear Detection System.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import json
from config import CFG
from models.faster_rcnn import FasterRCNN_Model
from models.mask_rcnn import MaskRCNN_Model
from models.fast_rcnn import FastRCNN_Model
from models.rcnn import RCNN
from utils.visualization import visualize_yolo_results
from utils.visualization import visualize_prediction, visualize_debug_images
from utils.evaluation import calculate_coco_map, validate
from utils.training import train_model
import glob

class SafetyGearDetector:
    """
    Class to detect safety gear using different Faster R-CNN models
    """

    def __init__(self, model_type='fasterrcnn_resnet50_fpn_v2', device=None, config=None):
        """
        Initialize detector

        Args:
            model_type (str): Model architecture to use
                             ('fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
                              'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn')
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config.copy() if config else CFG.__dict__.copy()
        self.config['model_type'] = model_type
        self.num_classes = len(self.config.get('CLASS_NAMES', []))
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on model type"""
        if 'maskrcnn' in self.model_type.lower():
            # For MaskRCNN models
            
            self.model = MaskRCNN_Model(
                num_classes=self.num_classes,  # Don't add +1, the model handles it
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif "fasterrcnn" in self.model_type.lower():
            # For FasterRCNN models (your existing code)
            self.model = FasterRCNN_Model(
                num_classes=self.num_classes,  # +1 for background
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif "fast_rcnn" in self.model_type.lower():
            # For FastRCNN models (your existing code)
            self.model = FastRCNN_Model(
                num_classes=self.num_classes,  # +1 for background
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif "rcnn" in self.model_type.lower():
            # For RCNN models (your existing code)
            self.model = RCNN(
                num_classes=self.num_classes,  # +1 for background
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif self.model_type.lower().startswith('rtdetr'):
            # For RT-DETR models
            from models.rt_detr import RTDETR_Model
            self.model = RTDETR_Model(
                num_classes=self.num_classes,
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif self.model_type.lower().startswith('detr'):
            # For DETR models
            from models.detr import DETR_Model
            self.model = DETR_Model(
                num_classes=self.num_classes,
                device=self.device,
                config=self.config  # config already contains model_type
            )
        elif self.model_type.lower().startswith(('yolov', 'yolo')):
            # For YOLO models (your existing code)
            from models.yolo import UltralyticsYOLO
            self.model = UltralyticsYOLO(
                num_classes=self.num_classes,
                model_type=self.model_type,
                device=self.device,
                config=self.config  # config already contains model_type
            )

    def predict(self, image, confidence_threshold=0.5, nms_threshold=0.3):
        """
        Run inference on an image

        Args:
            image: Either a numpy array, PIL Image, file path, or list of these
            confidence_threshold (float): Confidence threshold for predictions
            nms_threshold (float): NMS threshold for predictions

        Returns:
            tuple: (boxes, labels, scores, class_names)
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")

        # Process image input (could be path, numpy array, or PIL image)
        processed_image = self._process_image_input(image)
        if self.model_type.lower().startswith(('yolov', 'yolo')):
            # YOLO models handle inference differently
            results = self.model.predict(
                image,
                conf=confidence_threshold,
                iou=nms_threshold
            )
            
            # Process the results from YOLO format to our standard format
            boxes = []
            labels = []
            scores = []
            
            # Handle result (which is an Ultralytics Results object)
            for r in results:
                boxes.extend(r.boxes.xyxy.cpu().numpy())  # convert boxes to xyxy format
                labels.extend(r.boxes.cls.cpu().numpy().astype(int))
                scores.extend(r.boxes.conf.cpu().numpy())
            
            return np.array(boxes), np.array(labels), np.array(scores), self.config.get('CLASS_NAMES', [])
        else:
            # Run inference
            self.model.model.eval()
            with torch.no_grad():
                # Convert image to tensor and move to device
                if isinstance(processed_image, list):
                    # Handle batch of images
                    image_tensors = [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(self.device) for img in processed_image]
                    predictions = self.model.model(image_tensors)
                    
                    # Process each prediction
                    results = []
                    for pred in predictions:
                        # Get predictions above threshold
                        mask = pred['scores'] > confidence_threshold
                        boxes = pred['boxes'][mask].cpu().numpy()
                        labels = pred['labels'][mask].cpu().numpy()
                        scores = pred['scores'][mask].cpu().numpy()
                        
                        # Add to results
                        results.append((boxes, labels, scores, self.config.get('CLASS_NAMES', [])))
                    
                    return results
                else:
                    # Handle single image
                    image_tensor = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).to(self.device)
                    predictions = self.model.model([image_tensor])[0]
                    
                    # Get predictions above threshold
                    mask = predictions['scores'] > confidence_threshold
                    boxes = predictions['boxes'][mask].cpu().numpy()
                    labels = predictions['labels'][mask].cpu().numpy()
                    scores = predictions['scores'][mask].cpu().numpy()
                    
                    return boxes, labels, scores, self.config.get('CLASS_NAMES', [])

    def _process_image_input(self, image):
        """
        Process image input to numpy array

        Args:
            image: Either a numpy array, PIL Image, file path, or list of these

        Returns:
            np.ndarray: Processed image(s)
        """
        if isinstance(image, list):
            # Handle list of images
            return [self._process_single_image(img) for img in image]
        else:
            # Handle single image
            return self._process_single_image(image)

    def _process_single_image(self, image):
        """Process a single image to numpy array"""
        if isinstance(image, str):
            # Load image from file
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            image = np.array(image)
        
        # Ensure image is correctly formatted
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        return image

    def visualize_prediction(self, image, boxes=None, labels=None, scores=None, figsize=(12, 12)):
        """
        Visualize prediction on an image (delegate to utils.visualization)
        """
        return visualize_prediction(self, image, boxes, labels, scores, figsize)

    def visualize_debug_images(self, images, targets, outputs=None, max_images=2):
        """
        Visualize sample images with bounding boxes for debugging (delegate to utils.visualization)
        """
        return visualize_debug_images(self, images, targets, outputs, max_images)

    def save_model(self, filepath=None):
        """
        Save the model to disk

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        if filepath is None:
            filepath = os.path.join(
                self.config.get('OUTPUT_PATH', './'),
                f"{self.model_type}_safety_gear.pt"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Filter config to include only serializable types
        safe_config = {}
        for key, value in self.config.items():
            # Include only basic types that are guaranteed to be serializable
            if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
                # For nested dictionaries, also ensure they contain only serializable types
                if isinstance(value, dict):
                    # Recursive check for nested dictionaries (simplified)
                    try:
                        # Test if it can be JSON serialized (quick check for serializability)
                        json.dumps(value)
                        safe_config[key] = value
                    except (TypeError, OverflowError):
                        # If it can't be serialized, skip it
                        print(f"Warning: Skipping non-serializable config item '{key}'")
                else:
                    safe_config[key] = value
        
        # Save model with filtered config
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'config': safe_config  # Use the filtered config
        }, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        """
        Load a saved model from disk

        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Model file {filepath} does not exist")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update model type if it's in the checkpoint
        if 'model_type' in checkpoint:
            self.model_type = checkpoint['model_type']
        
        # Update num_classes if it's in the checkpoint
        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']
        
        # Initialize model with updated parameters
        self._initialize_model()
        
        # Load state dict
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update config if it's in the checkpoint
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Model loaded from {filepath}")
        return self

    def calculate_coco_map(self, data_loader):
        """Calculate COCO-style mAP (delegate to utils.evaluation)"""
        return calculate_coco_map(self, data_loader)

    def validate(self, data_loader):
        """Validate the model (delegate to utils.evaluation)"""
        return validate(self, data_loader)

    def train(self, train_loader, valid_loader=None, epochs=10, 
              lr=0.001, weight_decay=0.0005, batch_size=4, 
              fine_tune=False, freeze_backbone=True, unfreeze_layers=None,
              gradient_accumulation_steps=1):
         # For Fast R-CNN and R-CNN, use their custom training methods
        if self.model_type.lower() in ['fast_rcnn', 'rcnn']:
            print(f"Using custom training method for {self.model_type}")
            
            # Get datasets from the loaders
            train_dataset = train_loader.dataset
            valid_dataset = valid_loader.dataset if valid_loader else None
            
            # Use the model's own train method with the appropriate parameters
            return self.model.train(
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                pos_iou_threshold=0.5,  # Default values, you might want to make these configurable
                neg_iou_threshold=0.3,
                proposals_per_image=128,
                max_proposals=300
            )
        # For YOLO and RT-DETR models, training process is different
        elif self.model_type.startswith(('yolov4', 'yolov8', 'yolov12', 'rtdetr')):
            print(f"Training {self.model_type} with Ultralytics...")
            # Update config with training parameters
            training_params = {
                'data': 'data.yaml',  # Path to data.yaml file
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,  # Image size
                'device': self.device,
                'project': 'safety_gear_detection',
                'name': self.model_type,
                'lr0': lr,
                'weight_decay': weight_decay
            }
            
            # Train with Ultralytics
            results = self.model.train(**training_params)
            import glob
            val_images = glob.glob('./css-data/val/images/*.jpg')
            
            # Create output directory if it doesn't exist
            output_dir = f'./output/{self.model_type}_jpeg'
            os.makedirs(output_dir, exist_ok=True)
            
            # Run predictions on a few random samples
            n_samples = min(10, len(val_images))
            sample_images = np.random.choice(val_images, n_samples, replace=False)
            
            print("Running predictions on validation samples...")
            
            for img_path in sample_images:
                # Get the filename without path
                base_name = os.path.basename(img_path)
                
                # Use the Ultralytics model directly
                # This uses the native Ultralytics API
                self.model.model(
                    source=img_path, 
                    save=True,
                    project=output_dir,
                    name="",
                    exist_ok=True
                )
            
            print(f"Training completed. Sample predictions saved to {output_dir}")
            
            # Show some sample predictions
            prediction_images = glob.glob(f'{output_dir}/*.jpg')
            if prediction_images:
                n = min(5, len(prediction_images))
                
                from PIL import Image
                import matplotlib.pyplot as plt
                
                for i in range(n):
                    image = Image.open(prediction_images[i])
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title(f"Prediction {i+1}")
                    plt.show()
            else:
                print("No prediction images were found to display")
            
            return results
        else:
            # For Faster R-CNN and other models, use the common training infrastructure
            return train_model(self, train_loader, valid_loader, epochs, 
                            lr, weight_decay, batch_size, 
                            fine_tune, freeze_backbone, unfreeze_layers,
                            gradient_accumulation_steps)