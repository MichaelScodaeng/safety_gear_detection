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
from models.faster_rcnn import FasterRCNN_Model


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
        self.config = config or CFG.__dict__  # Use configuration from CFG if not provided
        self.num_classes = len(self.config.get('CLASS_NAMES', []))
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on model type"""
        # Determine the number of classes from the configuration
        num_classes = len(self.config.get('CLASS_NAMES', []))
        if num_classes == 0:
            raise ValueError("No classes specified in config")
        
        config = {
            'output_path': self.config.get('OUTPUT_PATH', './'),
            'model_type': self.model_type  # Pass the full model type name
        }
        
        # Create the Faster R-CNN model with the specified architecture
        supported_models = [
            'custom', 
            'fasterrcnn_resnet50_fpn', 
            'fasterrcnn_resnet50_fpn_v2',
            'fasterrcnn_mobilenet_v3_large_fpn', 
            'fasterrcnn_mobilenet_v3_large_320_fpn'
        ]
        
        if self.model_type in supported_models:
            self.model = FasterRCNN_Model(num_classes, self.device, config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {supported_models}")

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
        class_names = [self.config['CLASS_NAMES'][label] for label in labels]

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
            class_name = self.config['CLASS_NAMES'][label]
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
        plt.imshow(image_copy)
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
            os.makedirs(self.config['OUTPUT_PATH'], exist_ok=True)
            filepath = f"{self.config['OUTPUT_PATH']}/{self.model_type}_safety_gear.pt"

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

    

    def train(self, train_loader, valid_loader=None, epochs=10, 
          lr=0.001, weight_decay=0.0005, batch_size=4, 
          fine_tune=False, freeze_backbone=True, unfreeze_layers=None,
          gradient_accumulation_steps=1):
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
            unfreeze_layers (list): List of layer names to unfreeze during fine-tuning
            gradient_accumulation_steps (int): Number of steps to accumulate gradients

        Returns:
            dict: Training history
        """
        # Fine-tune if requested
        if fine_tune:
            self.model.fine_tune(
                freeze_backbone=freeze_backbone,
                unfreeze_layers=unfreeze_layers
            )
        
        # Create separate parameter groups with different learning rates
        params = []
        
        # For the replaced classification head, use a higher learning rate
        box_predictor_params = {
            "params": [p for n, p in self.model.model.named_parameters() 
                    if "box_predictor" in n and p.requires_grad],
            "lr": lr,
            "name": "box_predictor"
        }
        
        # For other trainable parameters, use a lower learning rate
        other_params = {
            "params": [p for n, p in self.model.model.named_parameters() 
                    if "box_predictor" not in n and p.requires_grad],
            "lr": lr * 0.1,  # 10x smaller learning rate
            "name": "backbone_features" 
        }
        
        # Define mixed precision training setup
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Add parameter groups
        params.append(box_predictor_params)
        if len(other_params["params"]) > 0:
            params.append(other_params)
        
        # Use AdamW optimizer with parameter groups
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Use learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Collect training arguments for printing
        training_args = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'fine_tune': fine_tune,
            'freeze_backbone': freeze_backbone,
            'unfreeze_layers': unfreeze_layers if unfreeze_layers else "None",
            'use_amp': torch.cuda.is_available(),
            'optimizer': optimizer
        }
        
        # Print hyperparameters
        try:
            self.model.print_hyperparameters(training_args)
        except AttributeError:
            print("Warning: print_hyperparameters method not found in model class")
            print(f"Training with: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': []
        }
        
        # Import tqdm for progress bar
        from tqdm import tqdm
        
        # Training loop
        for epoch in range(epochs):
            
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics
            epoch_loss = 0
            epoch_loss_classifier = 0
            epoch_loss_box_reg = 0
            epoch_loss_objectness = 0
            epoch_loss_rpn_box_reg = 0
            
            # Progress bar
            progress_bar = tqdm(train_loader, total=len(train_loader))
            
            # Initialize accumulation counter
            accumulated_steps = 0
            
            # Set model to training mode
            self.model.model.train()
            cnt = 0
            # Batch loop with debugging limit
            for images, targets in progress_bar:
                cnt += 1
                if cnt == 1:
                    #self.debug_batch(images, targets, batch_idx=cnt, print_full_model=True)
                    pass
                if cnt > 20:
                    break
                
                try:
                    # Move data to device
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Skip batches with no boxes
                    if any(len(t['boxes']) == 0 for t in targets):
                        continue
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Mixed precision forward pass if available
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            # Forward pass
                            loss_dict = self.model.model(images, targets)
                            
                            # Handle different return types (list or dict)
                            if isinstance(loss_dict, list):
                                # It's a list of dictionaries, use the first one
                                loss_dict = loss_dict[0] if loss_dict else {}
                            
                            # Calculate total loss
                            losses = sum(loss for loss in loss_dict.values()) if loss_dict else torch.tensor(0.0, device=self.device)
                            
                            # Scale for gradient accumulation
                            losses = losses / gradient_accumulation_steps
                        
                        # Backward pass with scaling
                        scaler.scale(losses).backward()
                    else:
                        # Standard forward pass
                        loss_dict = self.model.model(images, targets)
                        
                        # Handle different return types (list or dict)
                        if isinstance(loss_dict, list):
                            # It's a list of dictionaries, use the first one
                            loss_dict = loss_dict[0] if loss_dict else {}
                        
                        # Calculate total loss
                        losses = sum(loss for loss in loss_dict.values()) if loss_dict else torch.tensor(0.0, device=self.device)
                        
                        # Scale for gradient accumulation
                        losses = losses / gradient_accumulation_steps
                        
                        # Standard backward pass
                        losses.backward()
                    
                    # Update metrics (use original loss for logging)
                    actual_loss = losses.item() * gradient_accumulation_steps
                    epoch_loss += actual_loss
                    
                    # Update individual loss components if they exist
                    if isinstance(loss_dict, dict):
                        if 'loss_classifier' in loss_dict:
                            epoch_loss_classifier += loss_dict['loss_classifier'].item()
                        if 'loss_box_reg' in loss_dict:
                            epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
                        if 'loss_objectness' in loss_dict:
                            epoch_loss_objectness += loss_dict['loss_objectness'].item()
                        if 'loss_rpn_box_reg' in loss_dict:
                            epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
                    
                    # Update progress bar
                    progress_bar.set_description(f"Loss: {actual_loss:.4f}")
                    
                    # Increment accumulation counter
                    accumulated_steps += 1
                    
                    # Update weights after accumulating enough gradients
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        if use_amp:
                            # Update with scaler for mixed precision
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard update
                            optimizer.step()
                        
                        # Zero gradients
                        optimizer.zero_grad()
                
                except Exception as e:
                    print(f"Error in batch: {e}")
                    import traceback
                    traceback.print_exc()
                    # Skip problematic batch and continue
                    optimizer.zero_grad()
                    continue
            
            # Handle any remaining gradients
            if accumulated_steps % gradient_accumulation_steps != 0:
                if use_amp:
                    # Update with scaler for mixed precision
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard update
                    optimizer.step()
                
                # Zero gradients
                optimizer.zero_grad()
            
            # Calculate average losses (make sure cnt is used instead of len(train_loader))
            avg_loss = epoch_loss / len(train_loader)  # Using the counter instead of full dataset length
            avg_loss_classifier = epoch_loss_classifier / len(train_loader) if epoch_loss_classifier > 0 else 0
            avg_loss_box_reg = epoch_loss_box_reg / len(train_loader) if epoch_loss_box_reg > 0 else 0
            avg_loss_objectness = epoch_loss_objectness / len(train_loader) if epoch_loss_objectness > 0 else 0
            avg_loss_rpn_box_reg = epoch_loss_rpn_box_reg / len(train_loader) if epoch_loss_rpn_box_reg > 0 else 0
            
            # Print metrics
            print(f"Train Loss: {avg_loss:.4f}")
            print(f"  Classifier Loss: {avg_loss_classifier:.4f}")
            print(f"  Box Reg Loss: {avg_loss_box_reg:.4f}")
            print(f"  Objectness Loss: {avg_loss_objectness:.4f}")
            print(f"  RPN Box Reg Loss: {avg_loss_rpn_box_reg:.4f}")
            
            # Append to history
            history['train_loss'].append(avg_loss)
            
            # Validation
            if valid_loader:
                '''if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()'''
                
                # Use validate method for validation
                val_loss, val_map = self.validate(valid_loader)
                
                history['val_loss'].append(val_loss)
                history['val_map'].append(val_map)
                print(f"Val Loss  mAP@0.5 : {-val_loss:.4f}, mAP: {val_map:.4f}")
                
                # Update learning rate
                lr_scheduler.step(val_loss)
            
            # Gradually unfreeze deeper layers as training progresses (uncomment if needed)
            '''
            if epoch == int(epochs * 0.3):  # After 30% of epochs
                print("Unfreezing layer4 of the backbone")
                for name, param in self.model.model.named_parameters():
                    if "backbone.layer4" in name:
                        param.requires_grad = True
            
            elif epoch == int(epochs * 0.6):  # After 60% of epochs
                print("Unfreezing FPN layers")
                for name, param in self.model.model.named_parameters():
                    if "fpn" in name:
                        param.requires_grad = True
            '''
            
            # Save model checkpoint
            self.save_model(f"{self.config.get('OUTPUT_PATH', './')}/model_epoch_{epoch+1}.pt")
        
        return history

    def validate(self, data_loader):
        """
        Run validation using inference mode and calculate COCO mAP metrics
        """
        self.model.model.eval()
        
        # First, run inference on a single batch for visualization
        for images, targets in data_loader:
            try:
                # Move to device
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model.model(images)
                    
                # Visualize some examples
                print("\nVisualizing validation examples:")
                self.visualize_debug_images(images, targets, outputs, max_images=2)
                
                # Print shapes for debugging
                for i, (img, output, target) in enumerate(zip(images[:2], outputs[:2], targets[:2])):
                    if i == 0:
                        print("\nVisualizing validation examples:")
                        self.visualize_debug_images(images, targets, outputs)
                    
                    print(f"Image {i} shape: {img.shape}")
                    for k, v in output.items():
                        if hasattr(v, 'shape'):
                            print(f"  Output {k} shape: {v.shape}")
                    for k, v in target.items():
                        if hasattr(v, 'shape'):
                            print(f"  Target {k} shape: {v.shape}")
                
                # Just process one batch for visualization
                break
                
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                return float('inf'), 0.0
        
        # Calculate COCO-style mAP
        try:
            map_all, map_50, map_75 = self.calculate_coco_map(data_loader)
            
            # Return mAP@0.5 (commonly used) as validation metric
            # Use negative mAP as loss (since lower is better for optimizers)
            val_loss = -map_50
            val_map = map_all  # mAP@[0.5:0.95] is more comprehensive
            
            return val_loss, val_map
            
        except Exception as e:
            print(f"Error during mAP calculation: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), 0.0
    
    def validate_old(self, data_loader):
        """
        Run validation using only inference mode (no loss calculation)
        """
        self.model.model.eval()
        
        # Get a single batch for testing
        for images, targets in data_loader:
            try:
                # Move to device
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model.model(images)
                    
                # Print shapes for debugging
                for i, (img, output, target) in enumerate(zip(images[:2], outputs[:2], targets[:2])):
                    if i == 0:
                        print("\nVisualizing validation examples:")
                        self.visualize_debug_images(images, targets, outputs)
                    print(f"Image {i} shape: {img.shape}")
                    for k, v in output.items():
                        if hasattr(v, 'shape'):
                            print(f"  Output {k} shape: {v.shape}")
                    for k, v in target.items():
                        if hasattr(v, 'shape'):
                            print(f"  Target {k} shape: {v.shape}")
                
                # Calculate simple metrics
                total_preds = sum(len(output['scores']) for output in outputs)
                high_conf_preds = sum((output['scores'] > 0.5).sum().item() for output in outputs)
                
                print(f"Total predictions: {total_preds}")
                print(f"High confidence predictions: {high_conf_preds}")
                
                # Return placeholder metrics
                return 0.0, high_conf_preds / max(1, total_preds)
                
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                return float('inf'), 0.0
            
    def visualize_debug_images(self, images, targets, outputs=None, max_images=2):
        """
        Visualize sample images with bounding boxes for debugging purposes
        
        Args:
            images: List of image tensors
            targets: List of target dictionaries with ground truth boxes
            outputs: List of model prediction dictionaries (optional)
            max_images: Maximum number of images to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
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
                        class_name = self.config['CLASS_NAMES'][label-1] if label < len(self.config['CLASS_NAMES'])+1 else f"Class {label}"
                        ax1.text(
                            x1, y1-5, class_name, 
                            color='white', fontsize=8, 
                            bbox=dict(facecolor='g', alpha=0.8, pad=2)
                        )
                
                # Display predictions if available
                if outputs is not None:
                    output = outputs[i]
                    ax2.imshow(img_np)
                    
                    if 'boxes' in output and len(output['boxes']) > 0:
                        # Filter predictions by confidence
                        confidence_threshold = 0.5
                        if 'scores' in output:
                            mask = output['scores'] > confidence_threshold
                            boxes = output['boxes'][mask].cpu().numpy()
                            labels = output['labels'][mask].cpu().numpy() if 'labels' in output else [1] * len(boxes)
                            scores = output['scores'][mask].cpu().numpy() if 'scores' in output else [1.0] * len(boxes)
                        else:
                            boxes = output['boxes'].cpu().numpy()
                            labels = output['labels'].cpu().numpy() if 'labels' in output else [1] * len(boxes)
                            scores = [1.0] * len(boxes)
                        
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
                            class_name = self.config['CLASS_NAMES'][label-1] if label < len(self.config['CLASS_NAMES'])+1 else f"Class {label}"
                            ax2.text(
                                x1, y1-5, f"{class_name}: {score:.2f}", 
                                color='white', fontsize=8, 
                                bbox=dict(facecolor='r', alpha=0.8, pad=2)
                            )
                
                # Turn off axis labels
                ax1.axis('off')
                if outputs is not None:
                    ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Print some statistics about this image
                print(f"\nImage {i} statistics:")
                print(f"  Shape: {image.shape}")
                print(f"  Value range: [{img_np.min():.2f}, {img_np.max():.2f}]")
                
                if 'boxes' in target:
                    print(f"  Ground truth boxes: {len(target['boxes'])}")
                    
                    # Print box coordinate statistics
                    if len(target['boxes']) > 0:
                        boxes = target['boxes'].cpu().numpy()
                        print(f"  Box coordinate ranges:")
                        print(f"    x1: [{boxes[:, 0].min():.1f}, {boxes[:, 0].max():.1f}]")
                        print(f"    y1: [{boxes[:, 1].min():.1f}, {boxes[:, 1].max():.1f}]")
                        print(f"    x2: [{boxes[:, 2].min():.1f}, {boxes[:, 2].max():.1f}]")
                        print(f"    y2: [{boxes[:, 3].min():.1f}, {boxes[:, 3].max():.1f}]")
                        
                        # Check for fractional coordinates
                        has_fractional = np.any(boxes != np.floor(boxes))
                        print(f"  Has fractional coordinates: {has_fractional}")
                
                if outputs is not None and 'boxes' in output:
                    print(f"  Predicted boxes: {len(output['boxes'])}")
                    if 'scores' in output and len(output['scores']) > 0:
                        print(f"  Score range: [{output['scores'].min().item():.2f}, {output['scores'].max().item():.2f}]")
        
        except Exception as e:
            print(f"Error visualizing debug images: {e}")
            import traceback
            traceback.print_exc()
    def debug_batch(self, images, targets, predictions=None, batch_idx=0, print_full_model=False):
        """
        Print detailed information about a batch for debugging purposes with extensive shape printing
        
        Args:
            images: List of image tensors
            targets: List of target dictionaries
            predictions: Model predictions (optional)
            batch_idx: Batch index for reference
            print_full_model: Whether to print the full model structure
        """
        print("\n" + "="*80)
        print(f"BATCH {batch_idx} DEBUG INFORMATION - COMPREHENSIVE SHAPE ANALYSIS")
        print("="*80)
        
        # Image information
        print("\nIMAGE INFORMATION:")
        print(f"Number of images: {len(images)}")
        for i, img in enumerate(images[:2]):  # Only print first 2 images
            print(f"  Image {i}:")
            print(f"    Shape: {img.shape}")
            print(f"    Data type: {img.dtype}")
            print(f"    Device: {img.device}")
            print(f"    Value range: [{img.min().item():.2f}, {img.max().item():.2f}]")
            print(f"    Mean value: {img.mean().item():.2f}")
        
        # Target information
        print("\nTARGET INFORMATION:")
        print(f"Number of targets: {len(targets)}")
        for i, target in enumerate(targets[:2]):  # Only print first 2 targets
            print(f"  Target {i}:")
            for k, v in target.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: Shape {v.shape}, Data type {v.dtype}, Device {v.device}")
                    if k == 'boxes' and len(v) > 0:
                        for j, box in enumerate(v[:3]):  # Print first 3 boxes in detail
                            print(f"      Box {j}: {box.tolist()} (x1,y1,x2,y2)")
                        print(f"      Box range: [{v.min().item():.2f}, {v.max().item():.2f}]")
                    elif k == 'labels' and len(v) > 0:
                        print(f"      Labels: {v.tolist()}")
                else:
                    print(f"    {k}: {v}")
        
        # Prediction information (if available)
        if predictions is not None:
            print("\nPREDICTION INFORMATION:")
            print(f"Number of predictions: {len(predictions)}")
            for i, pred in enumerate(predictions[:2]):  # Only print first 2 predictions
                print(f"  Prediction {i}:")
                for k, v in pred.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: Shape {v.shape}, Data type {v.dtype}, Device {v.device}")
                        if k == 'boxes' and len(v) > 0:
                            for j, box in enumerate(v[:3]):  # Print first 3 boxes
                                print(f"      Box {j}: {box.tolist()} (x1,y1,x2,y2)")
                                if 'scores' in pred:
                                    print(f"      Score {j}: {pred['scores'][j].item():.4f}")
                        elif k == 'labels' and len(v) > 0:
                            print(f"      Labels: {v[:5].tolist()}")
                    else:
                        print(f"    {k}: {v}")
        
        # Print model structure (optional)
        if print_full_model:
            print("\nMODEL STRUCTURE:")
            print(self.model.model)
            
            # Print backbone structure
            if hasattr(self.model.model, 'backbone'):
                print("\nBACKBONE STRUCTURE:")
                print(self.model.model.backbone)
                
                # Print FPN
                if hasattr(self.model.model.backbone, 'fpn'):
                    print("\nFPN STRUCTURE:")
                    print(self.model.model.backbone.fpn)
            
            # Print RPN
            if hasattr(self.model.model, 'rpn'):
                print("\nRPN STRUCTURE:")
                print(self.model.model.rpn)
            
            # Print ROI heads
            if hasattr(self.model.model, 'roi_heads'):
                print("\nROI HEADS STRUCTURE:")
                print(self.model.model.roi_heads)
                
                # Print box predictor
                if hasattr(self.model.model.roi_heads, 'box_predictor'):
                    print("\nBOX PREDICTOR STRUCTURE:")
                    print(self.model.model.roi_heads.box_predictor)
        
        # Detailed forward pass test with shape tracking
        print("\nDETAILED FORWARD PASS TEST WITH SHAPE TRACKING:")
        try:
            # Save model state
            training = self.model.model.training
            # Set model to eval mode for testing
            self.model.model.eval()
            
            with torch.no_grad():
                # Clone images to avoid modifying them
                test_images = [img.clone().to(self.device) for img in images]
                
                # Print original image shapes
                print("  Original images shapes:")
                for i, img in enumerate(test_images):
                    print(f"    Image {i}: {img.shape}")
                
                # Track image transformation
                print("\n  Step 1: Image Transformation")
                if hasattr(self.model.model, 'transform'):
                    # Access the transform parameters
                    print(f"    Min size: {self.model.model.transform.min_size}")
                    print(f"    Max size: {self.model.model.transform.max_size}")
                    print(f"    Image mean: {self.model.model.transform.image_mean}")
                    print(f"    Image std: {self.model.model.transform.image_std}")
                    
                    # Apply transform manually to see the output sizes
                    transformed_images = self.model.model.transform(test_images)
                    
                    # Check the type of transformed_images
                    print(f"    Transformed images type: {type(transformed_images)}")
                    
                    # Handle different return types
                    if hasattr(transformed_images, 'tensors'):
                        # New style where transform returns an ImageList object
                        print(f"    Transformed images tensor shape: {transformed_images.tensors.shape}")
                        print(f"    Image sizes after transform: {transformed_images.image_sizes}")
                    elif isinstance(transformed_images, tuple):
                        # Old style or different return type where transform returns a tuple
                        print(f"    Transformed images is a tuple with {len(transformed_images)} elements")
                        for i, item in enumerate(transformed_images):
                            if isinstance(item, torch.Tensor):
                                print(f"      Tuple element {i}: Tensor of shape {item.shape}")
                            elif hasattr(item, 'shape'):
                                print(f"      Tuple element {i}: Shape {item.shape}")
                            else:
                                print(f"      Tuple element {i}: Type {type(item)}")
                    elif isinstance(transformed_images, torch.Tensor):
                        # Simple tensor return
                        print(f"    Transformed images tensor shape: {transformed_images.shape}")
                    else:
                        # Fallback for other return types
                        print(f"    Unknown transform return type: {type(transformed_images)}")
                
                # Track feature extraction (backbone)
                print("\n  Step 2: Feature Extraction (Backbone)")
                if hasattr(self.model.model, 'backbone'):
                    # Get transformed images
                    if hasattr(self.model.model, 'transform'):
                        transformed = self.model.model.transform(test_images)
                        
                        # Handle different transform return types
                        if isinstance(transformed, tuple):
                            # If transform returns a tuple, use the first element as input to backbone
                            # Adjust this based on your model's transform implementation
                            tensors_to_use = transformed[0] if isinstance(transformed[0], torch.Tensor) else transformed
                        elif hasattr(transformed, 'tensors'):
                            tensors_to_use = transformed.tensors
                        else:
                            tensors_to_use = transformed
                        
                        # Get features
                        features = self.model.model.backbone(tensors_to_use)
                        
                        if isinstance(features, torch.Tensor):
                            print(f"    Backbone output shape: {features.shape}")
                        else:
                            print(f"    Backbone output type: {type(features)}")
                            if isinstance(features, dict):
                                for k, v in features.items():
                                    print(f"      Feature map '{k}' shape: {v.shape}")
                            elif isinstance(features, tuple) or isinstance(features, list):
                                for i, feat in enumerate(features):
                                    if hasattr(feat, 'shape'):
                                        print(f"      Feature {i} shape: {feat.shape}")
                                    else:
                                        print(f"      Feature {i} type: {type(feat)}")
                
                # Track RPN output
                print("\n  Step 3: Region Proposal Network (RPN)")
                if hasattr(self.model.model, 'rpn'):
                    if hasattr(self.model.model, 'transform') and hasattr(self.model.model, 'backbone'):
                        transformed = self.model.model.transform(test_images)
                        
                        # Handle different transform return types
                        if isinstance(transformed, tuple):
                            tensors_to_use = transformed[0] if isinstance(transformed[0], torch.Tensor) else transformed
                            image_sizes = [transformed[1]] if len(transformed) > 1 else [(1000, 1000)]
                        elif hasattr(transformed, 'tensors') and hasattr(transformed, 'image_sizes'):
                            tensors_to_use = transformed.tensors
                            image_sizes = transformed.image_sizes
                        else:
                            tensors_to_use = transformed
                            image_sizes = [(1000, 1000)]
                        
                        # Get features
                        features = self.model.model.backbone(tensors_to_use)
                        
                        try:
                            # Get proposals
                            proposals, proposal_losses = self.model.model.rpn(
                                tensors_to_use, features, image_sizes, [{} for _ in range(len(test_images))]
                            )
                            
                            print(f"    Number of proposals: {len(proposals)}")
                            for i, props in enumerate(proposals[:2]):
                                print(f"      Proposals {i} shape: {props.shape}")
                        except Exception as e:
                            print(f"    Error during RPN inference: {e}")
                
                # Try inference mode (without targets)
                print("\n  Step 4: Full Forward Pass (Inference Mode)")
                inference_output = self.model.model(test_images)
                print(f"    Type: {type(inference_output)}")
                print(f"    First item type: {type(inference_output[0])}")
                print(f"    First item keys: {inference_output[0].keys()}")
                
                # Print each prediction's shape
                for i, pred in enumerate(inference_output[:2]):
                    print(f"    Prediction {i}:")
                    for k, v in pred.items():
                        if hasattr(v, 'shape'):
                            print(f"      {k}: Shape {v.shape}")
                            if k == 'boxes' and len(v) > 0:
                                print(f"        First box: {v[0].tolist()}")
                
                # Try training mode (with targets)
                print("\n  Step 5: Full Forward Pass (Training Mode)")
                # Format targets correctly
                test_targets = [{k: v.clone().to(self.device) for k, v in t.items()} for t in targets]
                
                # Print detailed target information
                print("    Target details:")
                for i, target in enumerate(test_targets[:2]):
                    print(f"      Target {i}:")
                    for k, v in target.items():
                        if hasattr(v, 'shape'):
                            print(f"        {k}: Shape {v.shape}")
                
                # Get training output
                training_output = self.model.model(test_images, test_targets)
                print(f"    Output type: {type(training_output)}")
                if isinstance(training_output, dict):
                    print(f"    Keys: {training_output.keys()}")
                    for k, v in training_output.items():
                        if hasattr(v, 'shape'):
                            print(f"      {k}: Shape {v.shape}, Value {v.item() if v.numel() == 1 else v}")
                        else:
                            print(f"      {k}: {v}")
                elif isinstance(training_output, list):
                    print(f"    Number of items: {len(training_output)}")
                    for i, item in enumerate(training_output[:2]):
                        print(f"      Item {i}:")
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if hasattr(v, 'shape'):
                                    print(f"        {k}: Shape {v.shape}, Value {v.item() if hasattr(v, 'item') else v}")
                                else:
                                    print(f"        {k}: {v}")
                else:
                    print(f"    Value: {training_output}")
            
            # Restore model state
            if training:
                self.model.model.train()
            else:
                self.model.model.eval()
                
        except Exception as e:
            print(f"  Error during detailed forward pass test: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*80 + "\n")

    def validate_with_simple_map(self, data_loader, iou_threshold=0.5):
        """
        Simplified mAP calculation
        """
        self.model.model.eval()
        
        # Track per-class metrics
        class_metrics = {i+1: {'true_positives': 0, 'false_positives': 0, 'ground_truths': 0} 
                        for i in range(self.num_classes)}
        
        with torch.no_grad():
            for images, targets in data_loader:
                # Move to device
                images = [img.to(self.device) for img in images]
                
                # Run inference
                outputs = self.model.model(images)
                
                # Process each image
                for img_idx, (output, target) in enumerate(zip(outputs, targets)):
                    # Get predictions
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()
                    
                    # Get ground truth
                    gt_boxes = target['boxes'].cpu().numpy()
                    gt_labels = target['labels'].cpu().numpy()
                    
                    # Mark which ground truths have been matched
                    gt_matched = [False] * len(gt_boxes)
                    
                    # Sort predictions by score (descending)
                    sorted_indices = np.argsort(-pred_scores)
                    
                    # Process each predicted box
                    for idx in sorted_indices:
                        pred_box = pred_boxes[idx]
                        pred_label = pred_labels[idx]
                        pred_score = pred_scores[idx]
                        
                        # Skip low confidence predictions
                        if pred_score < 0.5:
                            continue
                        
                        # Look for matching ground truth boxes of the same class
                        max_iou = 0
                        max_iou_idx = -1
                        
                        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                            # Skip already matched ground truths or different classes
                            if gt_matched[gt_idx] or gt_label != pred_label:
                                continue
                            
                            # Calculate IoU
                            iou = self._calculate_iou(pred_box, gt_box)
                            
                            if iou > max_iou:
                                max_iou = iou
                                max_iou_idx = gt_idx
                        
                        # If we found a match with high enough IoU
                        if max_iou > iou_threshold:
                            class_metrics[pred_label]['true_positives'] += 1
                            gt_matched[max_iou_idx] = True
                        else:
                            class_metrics[pred_label]['false_positives'] += 1
                    
                    # Count unmatched ground truths by class
                    for gt_label in gt_labels:
                        class_metrics[gt_label]['ground_truths'] += 1
        
        # Calculate precision and recall for each class
        precisions = []
        recalls = []
        
        for class_id, metrics in class_metrics.items():
            tp = metrics['true_positives']
            fp = metrics['false_positives']
            gt = metrics['ground_truths']
            
            # Calculate precision and recall
            precision = tp / max(tp + fp, 1)
            recall = tp / max(gt, 1)
            
            precisions.append(precision)
            recalls.append(recall)
            
            print(f"Class {class_id}: Precision={precision:.4f}, Recall={recall:.4f}, TP={tp}, FP={fp}, GT={gt}")
        
        # Calculate mean precision and recall
        mean_precision = sum(precisions) / max(len(precisions), 1)
        mean_recall = sum(recalls) / max(len(recalls), 1)
        
        # Calculate F1 score (harmonic mean of precision and recall)
        f1_score = 2 * mean_precision * mean_recall / max(mean_precision + mean_recall, 1e-6)
        
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        # Return negative F1 score as a "loss" and mean precision as a proxy for mAP
        return -f1_score, mean_precision

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Get coordinates of intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union
        union = box1_area + box2_area - intersection
        
        # Return IoU
        return intersection / max(union, 1e-6)
    
    def calculate_coco_map(self, data_loader):
        """
        Calculate COCO-style mAP for the model using multiple IoU thresholds (0.5:0.05:0.95)
        
        Args:
            data_loader: DataLoader for validation data
            
        Returns:
            tuple: (mAP@[0.5:0.05:0.95], mAP@0.5, mAP@0.75)
        """
        try:
            from pycocotools.cocoeval import COCOeval
            from pycocotools.coco import COCO
            import numpy as np
            import json
            import tempfile
            import os
        except ImportError:
            print("Error: pycocotools is required for COCO mAP calculation.")
            print("Please install it using: pip install pycocotools")
            return 0.0, 0.0, 0.0
        
        print("\nCalculating COCO-style mAP (0.5:0.05:0.95)...")
        self.model.model.eval()
        
        # Track all detections and ground truths
        all_predictions = []
        all_ground_truths = []
        image_id = 0
        annotation_id = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(data_loader)}")
                    
                # Process images and run inference
                processed_images = [img.to(self.device) for img in images]
                outputs = self.model.model(processed_images)
                
                # Store predictions and ground truths
                for img_idx, (output, target) in enumerate(zip(outputs, targets)):
                    current_image_id = image_id
                    image_id += 1
                    
                    # Skip images with no ground truth boxes
                    if 'boxes' not in target or len(target['boxes']) == 0:
                        continue
                    
                    # Get image dimensions for area calculation
                    img_height, img_width = processed_images[img_idx].shape[1:3]
                    
                    # Process predictions
                    if 'boxes' in output and len(output['boxes']) > 0:
                        boxes = output['boxes'].cpu().numpy()
                        scores = output['scores'].cpu().numpy()
                        labels = output['labels'].cpu().numpy()
                        
                        # Convert boxes to COCO format [x, y, width, height]
                        coco_boxes = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                            coco_boxes.append([float(x1), float(y1), float(width), float(height)])
                        
                        # Store each prediction
                        for box_idx, (box, score, label) in enumerate(zip(coco_boxes, scores, labels)):
                            prediction = {
                                'image_id': current_image_id,
                                'category_id': int(label),
                                'bbox': box,  # [x, y, width, height] format
                                'score': float(score)
                            }
                            all_predictions.append(prediction)
                    
                    # Process ground truths
                    if 'boxes' in target and len(target['boxes']) > 0:
                        boxes = target['boxes'].cpu().numpy()
                        labels = target['labels'].cpu().numpy()
                        
                        # Convert boxes to COCO format [x, y, width, height]
                        coco_boxes = []
                        areas = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                            coco_boxes.append([float(x1), float(y1), float(width), float(height)])
                            areas.append(float(width * height))
                        
                        # Store each ground truth
                        for box_idx, (box, label, area) in enumerate(zip(coco_boxes, labels, areas)):
                            ground_truth = {
                                'id': annotation_id,
                                'image_id': current_image_id,
                                'category_id': int(label),
                                'bbox': box,  # [x, y, width, height] format
                                'area': float(area),
                                'iscrowd': 0
                            }
                            all_ground_truths.append(ground_truth)
                            annotation_id += 1
        
        # Create temporary COCO-format files
        try:
            # Ground truth file
            gt_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
            gt_json = {
                'images': [{'id': i, 'width': 640, 'height': 640} for i in range(image_id)],
                'categories': [{'id': i+1, 'name': self.config['CLASS_NAMES'][i], 'supercategory': 'none'} 
                            for i in range(self.num_classes)],
                'annotations': all_ground_truths
            }
            json.dump(gt_json, gt_file)
            gt_file.close()
            
            # Prediction file
            pred_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
            json.dump(all_predictions, pred_file)
            pred_file.close()
            
            # Create COCO objects
            coco_gt = COCO(gt_file.name)
            
            # Check if we have any predictions
            if len(all_predictions) == 0:
                print("No predictions found, cannot calculate mAP.")
                map_all, map_50, map_75 = 0.0, 0.0, 0.0
            else:
                # Load predictions
                coco_dt = coco_gt.loadRes(pred_file.name)
                
                # Create COCO evaluator
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                
                # Set evaluation parameters
                coco_eval.params.imgIds = list(range(image_id))  # All images
                
                # FIX: Use standard 3-element maxDets as expected by COCOeval
                coco_eval.params.maxDets = [1, 10, 100]  # Standard COCO max detections
                
                # Run evaluation
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Extract mAP values
                # mAP@[0.5:0.95] (primary metric)
                map_all = coco_eval.stats[0]
                # mAP@0.5 (PASCAL VOC metric)
                map_50 = coco_eval.stats[1]
                # mAP@0.75 (strict metric)
                map_75 = coco_eval.stats[2]
                
                print(f"COCO mAP@[0.5:0.95]: {map_all:.4f}")
                print(f"COCO mAP@0.5: {map_50:.4f}")
                print(f"COCO mAP@0.75: {map_75:.4f}")
                
                # Individual class metrics
                if len(self.config['CLASS_NAMES']) > 1:
                    print("\nPer-class AP@[0.5:0.95]:")
                    # Get per-class AP
                    for category_idx, category_id in enumerate(coco_gt.getCatIds()):
                        if category_idx < len(self.config['CLASS_NAMES']):
                            try:
                                # Set evaluation for this class only
                                coco_eval.params.catIds = [category_id]
                                coco_eval.evaluate()
                                coco_eval.accumulate()
                                # Extract AP for this class - modified to handle index errors safely
                                precision = coco_eval.eval['precision']
                                if precision.shape[2] > category_idx:
                                    class_ap = precision[0, :, category_idx, 0, 2].mean()
                                    class_name = self.config['CLASS_NAMES'][category_idx]
                                    print(f"  {class_name}: {class_ap:.4f}")
                                else:
                                    print(f"  Class {category_idx+1}: No valid predictions")
                            except Exception as e:
                                print(f"  Error calculating metrics for class {category_idx+1}: {e}")
        
        except Exception as e:
            print(f"Error during mAP calculation: {e}")
            import traceback
            traceback.print_exc()
            map_all, map_50, map_75 = 0.0, 0.0, 0.0
        
        finally:
            # Clean up temporary files
            try:
                os.unlink(gt_file.name)
                os.unlink(pred_file.name)
            except Exception as e:
                print(f"Warning: Could not delete temporary files: {e}")
        
            return map_all, map_50, map_75