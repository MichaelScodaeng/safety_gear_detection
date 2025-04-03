"""
Implementation of DETR (Detection Transformer) from Hugging Face for Safety Gear Detection.
"""

import os
import torch
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.ops import box_convert

class DETR_Model:
    """
    Wrapper for DETR (DEtection TRansformer) model from Hugging Face.
    DETR uses a Transformer encoder-decoder architecture for end-to-end object detection.
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize DETR model

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config else {}
        self.model_type = config.get('model_type', 'detr')
        self.threshold = config.get('confidence_threshold', 0.5)
        self.model = None
        self.processor = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the DETR model from Hugging Face"""
        print(f"Initializing DETR model: {self.model_type}")
        
        # Map model type to Hugging Face model names
        model_map = {
            'detr': 'facebook/detr-resnet-50',
            'detr-resnet-50': 'facebook/detr-resnet-50',
            'detr-resnet-101': 'facebook/detr-resnet-101',
            'detr-r50-dc5': 'facebook/detr-resnet-50-dc5',
            'detr-r101-dc5': 'facebook/detr-resnet-101-dc5'
        }
        
        # Get model name based on model_type
        model_name = model_map.get(self.model_type, 'facebook/detr-resnet-50')
        
        try:
            # Load processor for preprocessing images
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            
            # Load pre-trained model
            pretrained_model = DetrForObjectDetection.from_pretrained(model_name)
            
            # Resize the classification head for our number of classes
            if self.num_classes != 91:  # COCO has 91 classes
                # Get the old classification head
                old_cls_head = pretrained_model.class_labels_classifier
                in_features = old_cls_head.in_features
                
                # Create a new head with our number of classes (+1 for background/no-object)
                new_cls_head = torch.nn.Linear(in_features, self.num_classes + 1)
                
                # Initialize the new head with the weights from the old head (for classes that overlap)
                with torch.no_grad():
                    # Copy weights for available classes (only up to our number of classes)
                    num_classes_to_copy = min(self.num_classes + 1, old_cls_head.out_features)
                    new_cls_head.weight.data[:num_classes_to_copy] = old_cls_head.weight.data[:num_classes_to_copy]
                    new_cls_head.bias.data[:num_classes_to_copy] = old_cls_head.bias.data[:num_classes_to_copy]
                
                # Replace the classification head
                pretrained_model.class_labels_classifier = new_cls_head
            
            self.model = pretrained_model
            self.model.to(self.device)
            print(f"DETR model loaded successfully: {model_name}")
            
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            raise

    def _convert_yolo_to_detr(self, yolo_annotations, image_shape):
        """
        Convert YOLOv8 annotations to DETR format.

        Args:
            yolo_annotations (list): List of YOLO annotations [class_id, x_center, y_center, width, height].
            image_shape (tuple): Shape of the image (height, width).

        Returns:
            dict: Converted annotations in DETR format.
        """
        h, w = image_shape
        boxes = []
        labels = []

        for annotation in yolo_annotations:
            class_id, x_center, y_center, width, height = annotation
            x_min = (x_center - width / 2) * w
            y_min = (y_center - height / 2) * h
            x_max = (x_center + width / 2) * w
            y_max = (y_center + height / 2) * h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(class_id))

        return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

    def predict(self, image, confidence_threshold=None, nms_threshold=None):
        """
        Run inference on an image

        Args:
            image: Image to run inference on (file path, PIL Image, or numpy array)
            confidence_threshold (float, optional): Confidence threshold
            nms_threshold (float, optional): NMS threshold (not used for DETR as it doesn't use NMS)

        Returns:
            tuple: (boxes, labels, scores, class_names)
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")
        
        threshold = confidence_threshold if confidence_threshold is not None else self.threshold
        
        # Convert image to PIL if it's a path or numpy array
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Prepare image for the model
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(self.device)
        
        # Forward pass through the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=threshold
        )[0]
        
        # Extract boxes, labels, and scores
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # Adjust labels to match the expected format (zero-indexed)
        labels = labels - 1
        
        # Get class names from config
        class_names = self.config.get('CLASS_NAMES', [])
        
        return boxes, labels, scores, class_names

    def train(self, train_loader, valid_loader=None, epochs=10, lr=0.0001, weight_decay=0.0001):
        """
        Train the DETR model

        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            epochs (int): Number of epochs to train
            lr (float): Learning rate
            weight_decay (float): Weight decay

        Returns:
            dict: Training history
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model first.")
        
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if valid_loader else None
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Process each batch
            for images, targets in train_loader:
                # Convert YOLO annotations to DETR format
                detr_targets = [
                    self._convert_yolo_to_detr(target["annotations"], image.shape[-2:])
                    for image, target in zip(images, targets)
                ]
                
                # Preprocess images
                encoding = self.processor(images=images, annotations=detr_targets, return_tensors="pt")
                pixel_values = encoding["pixel_values"].to(self.device)
                labels = encoding["labels"]
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        return history

    def _validate(self, dataloader):
        """Calculate validation loss"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                # Convert targets to DETR format
                detr_targets = self._convert_targets_to_detr_format(targets)
                
                # Preprocess images
                encoding = self.processor(images=images, annotations=detr_targets, return_tensors="pt")
                pixel_values = encoding["pixel_values"].to(self.device)
                labels = encoding["labels"]
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Update metrics
                val_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return val_loss / num_batches

    def _convert_targets_to_detr_format(self, targets):
        """Convert targets to DETR format"""
        detr_targets = []
        
        for target in targets:
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            
            # Convert to COCO format where class 0 is background
            labels = labels
            
            # Convert to [center_x, center_y, width, height] format
            boxes_cxcywh = box_convert(torch.tensor(boxes), 'xyxy', 'cxcywh').numpy()
            
            annotations = []
            for i, (box, label) in enumerate(zip(boxes_cxcywh, labels)):
                annotations.append({
                    'bbox': box,
                    'category_id': int(label),
                    'id': i
                })
            
            detr_targets.append({
                'annotations': annotations,
                'image_id': 0  # Dummy value
            })
        
        return detr_targets

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
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'config': self.config
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
        
        # Initialize model with updated parameters if not already done
        if self.model is None:
            self._initialize_model()
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update config if it's in the checkpoint
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Model loaded from {filepath}")
        return self
