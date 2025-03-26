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

    def train(self, train_loader, valid_loader=None, epochs=10, lr=0.001, weight_decay=0.0005, batch_size=4):
        """
        Train the Faster R-CNN model

        Args:
            train_loader (DataLoader): DataLoader for training
            valid_loader (DataLoader, optional): DataLoader for validation
            epochs (int): Number of epochs to train
            lr (float): Learning rate
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

        # No need to create data loaders since they are already provided
        # Rest of the method can remain the same
        # ...

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=4
        )

        if valid_dataset:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
                num_workers=4
            )

            # Set model to training mode
        self.model.train()

        # Create optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Training history
        history = {
            'train_loss': [],
            'train_loss_classifier': [],
            'train_loss_box_reg': [],
            'train_loss_objectness': [],
            'train_loss_rpn_box_reg': [],
            'val_map': [] if valid_loader else None
        }

        # Training loop
        for epoch in range(epochs):
            # Set model to training mode
            self.model.train()

            # Track losses
            epoch_loss = 0
            epoch_loss_classifier = 0
            epoch_loss_box_reg = 0
            epoch_loss_objectness = 0
            epoch_loss_rpn_box_reg = 0

            # Start time
            start_time = time.time()

            # Iterate through training data
            for i, (images, targets) in enumerate(train_loader):
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = self.model(images, targets)

                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Update epoch loss
                epoch_loss += losses.item()
                epoch_loss_classifier += loss_dict['loss_classifier'].item()
                epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
                epoch_loss_objectness += loss_dict['loss_objectness'].item()
                epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, "
                          f"Loss: {losses.item():.4f}, "
                          f"Class Loss: {loss_dict['loss_classifier'].item():.4f}, "
                          f"Box Reg Loss: {loss_dict['loss_box_reg'].item():.4f}, "
                          f"Objectness Loss: {loss_dict['loss_objectness'].item():.4f}, "
                          f"RPN Box Reg Loss: {loss_dict['loss_rpn_box_reg'].item():.4f}")

            # Update learning rate
            lr_scheduler.step()

            # Calculate average loss
            epoch_loss /= len(train_loader)
            epoch_loss_classifier /= len(train_loader)
            epoch_loss_box_reg /= len(train_loader)
            epoch_loss_objectness /= len(train_loader)
            epoch_loss_rpn_box_reg /= len(train_loader)

            # Update history
            history['train_loss'].append(epoch_loss)
            history['train_loss_classifier'].append(epoch_loss_classifier)
            history['train_loss_box_reg'].append(epoch_loss_box_reg)
            history['train_loss_objectness'].append(epoch_loss_objectness)
            history['train_loss_rpn_box_reg'].append(epoch_loss_rpn_box_reg)

            # Validation
            if valid_dataset:
                # Set model to evaluation mode
                self.model.eval()

                # Calculate mAP on validation set
                mAP = self.evaluate(valid_loader)['mAP']
                history['val_map'].append(mAP)

                # Print results
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Val mAP: {mAP:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

                # Save best model
                if mAP > history.get('best_map', 0):
                    history['best_map'] = mAP
                    self.save_model(os.path.join(self.config.get('output_path', '.'), 'best_model.pt'))
            else:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

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
