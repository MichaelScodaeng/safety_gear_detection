"""
Implementation of Faster R-CNN algorithm for Safety Gear Detection.
"""

import os
import time
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image

from .base import RCNNBase

import os
import time
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image

from .base import RCNNBase


class FasterRCNN_Model(RCNNBase):
    """
    Implementation of Faster R-CNN
    Faster R-CNN improves on Fast R-CNN by:
    1. Replacing the selective search algorithm with a Region Proposal Network (RPN)
    2. Training the RPN and detector in an end-to-end fashion
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize Faster R-CNN

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        super().__init__(num_classes, device, config)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Faster R-CNN model with a pre-trained backbone"""
        # Using torchvision's implementation for simplicity
        # Load a pre-trained backbone (ResNet-50 + FPN)
        backbone = resnet_fpn_backbone(
            'resnet50', weights="DEFAULT"
        )

        # Create anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # Create ROI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create Faster R-CNN model
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=self.num_classes + 1,  # +1 for background class
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=600,
            max_size=1000,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100
        )

        # Move model to device
        self.model.to(self.device)

    def fine_tune(self, model_path=None, freeze_backbone=True, train_rpn_only=False):
        """
        Fine-tune the model

        Args:
            model_path (str, optional): Path to pre-trained model
            freeze_backbone (bool): Whether to freeze the backbone
            train_rpn_only (bool): Whether to train only the RPN
        """
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.load_model(model_path)

        # Freeze backbone layers if requested
        if freeze_backbone:
            # Freeze backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Train only RPN if requested
        if train_rpn_only:
            # Freeze all layers except RPN
            for name, param in self.model.named_parameters():
                if "rpn" not in name:
                    param.requires_grad = False

    def train(self, train_loader, valid_loader=None, epochs=10, lr=0.0001, weight_decay=0.0005, batch_size=2):
        """
        Train the Faster R-CNN model with robust error handling

        Args:
            train_loader (DataLoader): DataLoader for training
            valid_loader (DataLoader, optional): DataLoader for validation
            epochs (int): Number of epochs to train
            lr (float): Learning rate (reduced from 0.001)
            weight_decay (float): Weight decay for optimizer
            batch_size (int): Batch size for training

        Returns:
            dict: Training history
        """
        # Initialize the model if not done yet
        if self.model is None:
            self._initialize_model()

        # Move model to device
        self.model.to(self.device)

        # Set model to training mode
        self.model.train()

        # Create optimizer - try Adam instead of SGD for better stability
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        # Alternative: Use SGD with lower learning rate
        # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        # Use ReduceLROnPlateau scheduler instead of StepLR for adaptive learning rate
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'train_loss_classifier': [],
            'train_loss_box_reg': [],
            'train_loss_objectness': [],
            'train_loss_rpn_box_reg': [],
            'val_map': [] if valid_loader else None,
            'best_map': 0
        }

        # Training loop
        nan_count = 0  # Count NaN occurrences to detect persistent issues
        for epoch in range(epochs):
            # Set model to training mode
            self.model.train()

            # Track losses
            epoch_loss = 0
            epoch_loss_classifier = 0
            epoch_loss_box_reg = 0
            epoch_loss_objectness = 0
            epoch_loss_rpn_box_reg = 0
            valid_batches = 0  # Count valid batches

            # Start time
            start_time = time.time()

            # Iterate through training data with error handling
            for i, (images, targets) in enumerate(train_loader):
                try:
                    # Skip empty batches
                    if any(len(t['boxes']) == 0 for t in targets):
                        print(f"Skipping batch {i + 1} with empty targets")
                        continue

                    # Move data to device
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # Verify boxes have proper dimensions
                    skip_batch = False
                    for t in targets:
                        # Check for invalid boxes
                        if t['boxes'].size(0) > 0:
                            widths = t['boxes'][:, 2] - t['boxes'][:, 0]
                            heights = t['boxes'][:, 3] - t['boxes'][:, 1]
                            if torch.any(widths <= 0) or torch.any(heights <= 0):
                                print(f"Found invalid box dimensions in batch {i + 1}, skipping")
                                skip_batch = True
                                break

                    if skip_batch:
                        continue

                    # Forward pass
                    loss_dict = self.model(images, targets)

                    # Check for NaN values in loss
                    if any(torch.isnan(loss).item() for loss in loss_dict.values()):
                        nan_count += 1
                        print(f"NaN detected in batch {i + 1}. Skipping... (Total NaN incidents: {nan_count})")

                        # If too many NaNs, reduce learning rate
                        if nan_count % 5 == 0:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                                print(f"Reducing learning rate to {param_group['lr']}")

                        continue

                    # Calculate total loss
                    losses = sum(loss for loss in loss_dict.values())

                    # Skip abnormally high losses that might lead to NaN
                    if losses.item() > 50:
                        print(f"Extremely high loss ({losses.item():.2f}) in batch {i + 1}. Skipping...")
                        continue

                    # Backward pass
                    optimizer.zero_grad()
                    losses.backward()

                    # Add gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                    optimizer.step()

                    # Update epoch loss
                    epoch_loss += losses.item()
                    epoch_loss_classifier += loss_dict['loss_classifier'].item()
                    epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
                    epoch_loss_objectness += loss_dict['loss_objectness'].item()
                    epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
                    valid_batches += 1

                    # Print progress
                    if (i + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, "
                              f"Loss: {losses.item():.4f}, "
                              f"Class Loss: {loss_dict['loss_classifier'].item():.4f}, "
                              f"Box Reg Loss: {loss_dict['loss_box_reg'].item():.4f}, "
                              f"Objectness Loss: {loss_dict['loss_objectness'].item():.4f}, "
                              f"RPN Box Reg Loss: {loss_dict['loss_rpn_box_reg'].item():.4f}")

                except Exception as e:
                    print(f"Error in batch {i + 1}: {str(e)}")
                    continue

            # Check if training made progress
            if valid_batches == 0:
                print("No valid batches in this epoch. Check your data or reduce learning rate further.")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    print(f"Reducing learning rate to {param_group['lr']}")
                continue

            # Calculate average loss
            epoch_loss /= valid_batches
            epoch_loss_classifier /= valid_batches
            epoch_loss_box_reg /= valid_batches
            epoch_loss_objectness /= valid_batches
            epoch_loss_rpn_box_reg /= valid_batches

            # Update history
            history['train_loss'].append(epoch_loss)
            history['train_loss_classifier'].append(epoch_loss_classifier)
            history['train_loss_box_reg'].append(epoch_loss_box_reg)
            history['train_loss_objectness'].append(epoch_loss_objectness)
            history['train_loss_rpn_box_reg'].append(epoch_loss_rpn_box_reg)

            # Update learning rate based on training loss
            lr_scheduler.step(epoch_loss)

            # Validation
            if valid_loader:
                # Set model to evaluation mode
                self.model.eval()

                # Calculate mAP on validation set
                val_metrics = self.evaluate(valid_loader)
                mAP = val_metrics['mAP']
                history['val_map'].append(mAP)

                # Print results
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Val mAP: {mAP:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

                # Save best model
                if mAP > history['best_map']:
                    history['best_map'] = mAP
                    self.save_model(os.path.join(self.config.get('output_path', '.'), 'best_model.pt'))
            else:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

                # Save checkpoint every epoch
                self.save_model(os.path.join(self.config.get('output_path', '.'), f'checkpoint_epoch_{epoch + 1}.pt'))

        return history

    def predict(self, image, confidence_threshold=0.5, nms_threshold=0.3):
        """
        Run inference on an image

        Args:
            image: Image to run inference on (numpy array or PIL image)
            confidence_threshold (float): Confidence threshold for detections
            nms_threshold (float): NMS threshold

        Returns:
            tuple: (boxes, labels, scores)
        """
        # Initialize model if not done yet
        if self.model is None:
            self._initialize_model()

        # Set model to evaluation mode
        self.model.eval()

        # Convert image to the right format
        if isinstance(image, str):
            # Load image from path
            if os.path.exists(image):
                image = Image.open(image).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {image}")

        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            transform = self._get_transform(train=False)
            transformed = transform(image=image, bboxes=[], labels=[])
            image_tensor = transformed['image']
        elif isinstance(image, Image.Image):
            # Convert PIL image to tensor
            transform = self._get_transform(train=False)
            image_np = np.array(image)
            transformed = transform(image=image_np, bboxes=[], labels=[])
            image_tensor = transformed['image']
        elif isinstance(image, torch.Tensor):
            # Already a tensor
            image_tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Add batch dimension and move to device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            # torchvision's Faster R-CNN returns a list of dictionaries in eval mode
            prediction = self.model(image_tensor)[0]

            # Get predictions
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()

            # Filter by confidence threshold
            keep = scores >= confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Convert class indices back to original indices (undo the +1 that was added for background)
            labels = labels - 1

        return boxes, labels, scores

    def visualize_prediction(self, image, boxes=None, labels=None, scores=None, figsize=(12, 12)):
        """
        Visualize predictions on an image

        Args:
            image: Image to visualize (numpy array or PIL image)
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            labels: Class labels
            scores: Confidence scores
            figsize: Figure size

        Returns:
            Image with bounding boxes drawn
        """
        from models.base import RCNNBase
        # Use the base class implementation
        return super().visualize_prediction(image, boxes, labels, scores, figsize)