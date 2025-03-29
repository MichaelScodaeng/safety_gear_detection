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
    def train(self, train_loader, val_loader=None, epochs=10, optimizer=None, 
          lr_scheduler=None, batch_size=4, gradient_accumulation_steps=1,
          fine_tune=False, freeze_backbone=True, unfreeze_layers=None):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            optimizer: Optimizer to use (if None, AdamW will be created)
            lr_scheduler: Learning rate scheduler
            batch_size: Batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            fine_tune: Whether to apply fine-tuning
            freeze_backbone: Whether to freeze backbone during fine-tuning
            unfreeze_layers: Which layers to unfreeze
            
        Returns:
            dict: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Create optimizer if none provided
        if optimizer is None:
            # Only optimize parameters that require gradients
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)
        
        # Create scheduler if none provided
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=3
            )
        
        # Determine if we can use mixed precision
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Collect training arguments for printing
        training_args = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'fine_tune': fine_tune,
            'freeze_backbone': freeze_backbone,
            'unfreeze_layers': unfreeze_layers if unfreeze_layers else "None",
            'use_amp': use_amp,
            'optimizer': optimizer
        }
        
        # Print hyperparameters
        self.print_hyperparameters(training_args)
    
    # Rest of your training implementation...
        
        # Create optimizer if none provided
        if optimizer is None:
            # Only optimize parameters that require gradients
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)
        
        # Create scheduler if none provided
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=3, 
                verbose=True
            )
        
        # Import tqdm for progress bar
        from tqdm import tqdm
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': []
        }
        
        # For mAP calculation
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            mAP_metric = MeanAveragePrecision()
        except ImportError:
            print("torchmetrics not found. mAP will not be calculated.")
            mAP_metric = None
        
        # Best model tracking
        best_map = 0
        patience = 5
        patience_counter = 0
        
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
            
            # Batch loop
            for images, targets in progress_bar:
                # Move data to device
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Skip batches with no boxes
                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                    
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                losses.backward()
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                epoch_loss += losses.item()
                if 'loss_classifier' in loss_dict:
                    epoch_loss_classifier += loss_dict['loss_classifier'].item()
                if 'loss_box_reg' in loss_dict:
                    epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
                if 'loss_objectness' in loss_dict:
                    epoch_loss_objectness += loss_dict['loss_objectness'].item()
                if 'loss_rpn_box_reg' in loss_dict:
                    epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
                
                # Update progress bar
                progress_bar.set_description(f"Loss: {losses.item():.4f}")
            
            # Calculate average losses
            avg_loss = epoch_loss / len(train_loader)
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
            if val_loader:
                val_loss, val_map = self.validate(val_loader, mAP_metric)
                history['val_loss'].append(val_loss)
                history['val_map'].append(val_map)
                print(f"Val Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
                
                # Update learning rate
                lr_scheduler.step(val_loss)
                
                # Checkpointing
                if val_map > best_map:
                    best_map = val_map
                    patience_counter = 0
                    # Save best model
                    self.save_model(f"{self.config.get('output_path', './')}/best_model.pt")
                    print(f"New best model saved with mAP: {val_map:.4f}")
                else:
                    patience_counter += 1
                    print(f"mAP did not improve. Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Gradually unfreeze deeper layers as training progresses
            if epoch == int(epochs * 0.3):  # After 30% of epochs
                print("Unfreezing layer4 of the backbone")
                for name, param in self.model.named_parameters():
                    if "backbone.layer4" in name:
                        param.requires_grad = True
            
            elif epoch == int(epochs * 0.6):  # After 60% of epochs
                print("Unfreezing FPN layers")
                for name, param in self.model.named_parameters():
                    if "fpn" in name:
                        param.requires_grad = True
            
            # Save model checkpoint for this epoch
            self.save_model(f"{self.config.get('output_path', './')}/model_epoch_{epoch+1}.pt")
        
        return history

    def validate(self, data_loader, mAP_metric=None):
        """Validate the model on a validation dataset"""
        self.model.eval()
        val_loss = 0
        
        if mAP_metric is None:
            try:
                from torchmetrics.detection.mean_ap import MeanAveragePrecision
                mAP_metric = MeanAveragePrecision()
            except ImportError:
                print("torchmetrics not found. mAP will not be calculated.")
                mAP_metric = None
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Skip batches with no boxes
                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                    
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Get predictions for mAP calculation
                if mAP_metric is not None:
                    self.model.eval()
                    predictions = self.model(images)
                    mAP_metric.update(predictions, targets)
        
        # Calculate metrics
        val_map = 0
        if mAP_metric is not None:
            metric_results = mAP_metric.compute()
            val_map = metric_results['map'].item()
            mAP_metric.reset()
        
        # Return average loss and mAP
        return val_loss / len(data_loader), val_map