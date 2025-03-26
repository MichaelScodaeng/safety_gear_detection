"""
Implementation of the original R-CNN algorithm for Safety Gear Detection.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from torchvision.ops import nms
from PIL import Image

from .base import RCNNBase

class RCNN(RCNNBase):
    """
    Implementation of the original R-CNN
    R-CNN uses selective search for region proposals and a CNN for feature extraction
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize R-CNN

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        super().__init__(num_classes, device, config)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the R-CNN model with a pre-trained backbone"""
        # Load a pre-trained ResNet model for feature extraction
        backbone = torchvision.models.resnet50(weights="DEFAULT")

        # Remove the last fully connected layer
        modules = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        # Create classifier
        in_features = backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes + 1)  # +1 for background class
        )

        # Create bounding box regressor
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4 * self.num_classes)  # (tx, ty, tw, th) for each class
        )

        # Combine into a single model
        self.model = nn.ModuleDict({
            'feature_extractor': self.feature_extractor,
            'classifier': self.classifier,
            'bbox_regressor': self.bbox_regressor
        })

        # Move model to device
        self.model.to(self.device)

    def _generate_proposals(self, image, max_proposals=2000):
        """
        Generate region proposals using Selective Search

        Args:
            image (np.ndarray): Input image
            max_proposals (int): Maximum number of proposals to generate

        Returns:
            np.ndarray: Region proposals in [x1, y1, x2, y2] format
        """
        # Configure selective search
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()

        # Run selective search
        rects = self.ss.process()

        # Convert to [x1, y1, x2, y2] format
        proposals = []
        for (x, y, w, h) in rects[:max_proposals]:
            proposals.append([x, y, x + w, y + h])

        return np.array(proposals, dtype=np.float32)

    def _extract_features(self, image, proposals):
        """
        Extract features for each proposal

        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            proposals (np.ndarray): Region proposals in [x1, y1, x2, y2] format

        Returns:
            torch.Tensor: Features for each proposal
        """
        # Convert image back to numpy if needed
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_np = img_np.astype(np.uint8)
        else:
            img_np = image

        # Extract regions
        regions = []
        for x1, y1, x2, y2 in proposals:
            # Ensure coordinates are within image boundaries
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_np.shape[1], int(x2)), min(img_np.shape[0], int(y2))

            # Skip if region is too small
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract region
            region = img_np[y1:y2, x1:x2]

            # Resize to match model input size
            region = cv2.resize(region, (224, 224))

            # Convert to tensor
            region = TF.to_tensor(region)
            region = TF.normalize(region, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            regions.append(region)

        # Stack regions into a batch
        if not regions:
            return torch.zeros((0, 2048), device=self.device)

        regions_batch = torch.stack(regions).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(regions_batch)

        return features

    def train(self, train_dataset, valid_dataset=None, epochs=10, lr=0.001, weight_decay=0.0005, batch_size=32,
              pos_iou_threshold=0.5, neg_iou_threshold=0.3, proposals_per_image=128, max_proposals=300):
        """
        Train the RCNN model

        Args:
            train_dataset (Dataset): Dataset for training
            valid_dataset (Dataset, optional): Dataset for validation
            epochs (int): Number of epochs to train
            lr (float): Learning rate
            weight_decay (float): Weight decay for optimizer
            batch_size (int): Batch size for training
            pos_iou_threshold (float): IoU threshold for positive samples
            neg_iou_threshold (float): IoU threshold for negative samples
            proposals_per_image (int): Number of proposals to use per image
            max_proposals (int): Maximum number of proposals to generate

        Returns:
            dict: Training history
        """
        # Initialize the model if not done yet
        if self.model is None:
            self._initialize_model()

        # Move model to device
        self.model.to(self.device)

        # Create optimizers
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Loss functions
        classification_loss_fn = nn.CrossEntropyLoss()
        regression_loss_fn = nn.SmoothL1Loss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if valid_dataset else None,
            'val_map': [] if valid_dataset else None
        }

        # Training loop
        for epoch in range(epochs):
            # Set model to training mode
            self.model.train()

            # Track losses
            epoch_loss = 0
            epoch_cls_loss = 0
            epoch_reg_loss = 0

            # Track time
            start_time = time.time()

            # Iterate through dataset
            for idx in range(len(train_dataset)):
                # Get data
                image, target = train_dataset[idx]
                image_np = image.numpy().transpose(1, 2, 0)
                image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                image_np = image_np.astype(np.uint8)

                # Generate proposals
                proposals = self._generate_proposals(image_np, max_proposals=max_proposals)

                # Match proposals to ground truth
                gt_boxes = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()

                # Skip if no ground truth
                if len(gt_boxes) == 0:
                    continue

                # Calculate IoU between proposals and ground truth
                ious = np.zeros((len(proposals), len(gt_boxes)))
                for i, proposal in enumerate(proposals):
                    for j, gt_box in enumerate(gt_boxes):
                        ious[i, j] = self._calculate_iou(proposal, gt_box)

                # Get maximum IoU for each proposal
                max_ious = np.max(ious, axis=1)
                gt_argmax = np.argmax(ious, axis=1)

                # Assign labels to proposals
                proposal_labels = np.zeros(len(proposals), dtype=np.int64)
                proposal_boxes = np.zeros((len(proposals), 4), dtype=np.float32)

                # Assign positive samples
                pos_indices = np.where(max_ious >= pos_iou_threshold)[0]
                proposal_labels[pos_indices] = gt_labels[gt_argmax[pos_indices]]

                # Assign negative samples
                neg_indices = np.where(max_ious < neg_iou_threshold)[0]
                proposal_labels[neg_indices] = 0  # 0 is background

                # Assign regression targets
                for i in pos_indices:
                    gt_box = gt_boxes[gt_argmax[i]]
                    proposal_box = proposals[i]

                    # Calculate regression targets
                    tx = (gt_box[0] - proposal_box[0]) / (proposal_box[2] - proposal_box[0])
                    ty = (gt_box[1] - proposal_box[1]) / (proposal_box[3] - proposal_box[1])
                    tw = np.log((gt_box[2] - gt_box[0]) / (proposal_box[2] - proposal_box[0]))
                    th = np.log((gt_box[3] - gt_box[1]) / (proposal_box[3] - proposal_box[1]))

                    proposal_boxes[i] = np.array([tx, ty, tw, th])

                # Limit number of proposals for training
                if len(pos_indices) > proposals_per_image // 2:
                    pos_indices = np.random.choice(pos_indices, proposals_per_image // 2, replace=False)

                if len(neg_indices) > proposals_per_image - len(pos_indices):
                    neg_indices = np.random.choice(neg_indices, proposals_per_image - len(pos_indices), replace=False)

                # Combine positive and negative indices
                keep_indices = np.concatenate([pos_indices, neg_indices])

                # Keep only selected proposals
                proposals = proposals[keep_indices]
                proposal_labels = proposal_labels[keep_indices]
                proposal_boxes = proposal_boxes[keep_indices]

                # Skip if no proposals
                if len(proposals) == 0:
                    continue

                # Extract features
                features = self._extract_features(image, proposals)

                # Skip if no features
                if len(features) == 0:
                    continue

                # Forward pass
                class_scores = self.model['classifier'](features)
                regression_pred = self.model['bbox_regressor'](features)

                # Compute losses
                cls_loss = classification_loss_fn(class_scores, torch.tensor(proposal_labels, device=self.device))

                # Compute regression loss only for positive samples
                pos_mask = proposal_labels > 0
                if np.any(pos_mask):
                    # Get regression targets for positive samples
                    pos_indices = np.where(pos_mask)[0]
                    pos_labels = proposal_labels[pos_mask] - 1  # -1 because background is 0

                    # Get predicted regression values for positive samples
                    reg_pred = []
                    for i, label in enumerate(pos_labels):
                        idx = label * 4
                        reg_pred.append(regression_pred[pos_indices[i], idx:idx+4])

                    reg_pred = torch.stack(reg_pred)

                    # Get ground truth regression values
                    reg_target = torch.tensor(proposal_boxes[pos_mask], device=self.device)

                    # Compute regression loss
                    reg_loss = regression_loss_fn(reg_pred, reg_target)
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)

                # Combine losses
                loss = cls_loss + reg_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()

                # Print progress
                if (idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {idx+1}/{len(train_dataset)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Cls Loss: {cls_loss.item():.4f}, "
                          f"Reg Loss: {reg_loss.item():.4f}")

            # Update learning rate
            lr_scheduler.step()

            # Calculate average loss
            epoch_loss /= len(train_dataset)
            epoch_cls_loss /= len(train_dataset)
            epoch_reg_loss /= len(train_dataset)

            history['train_loss'].append(epoch_loss)

            # Validation
            if valid_dataset is not None:
                # Set model to evaluation mode
                self.model.eval()

                val_loss = 0
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    for idx in range(len(valid_dataset)):
                        # Get data
                        image, target = valid_dataset[idx]
                        image_np = image.numpy().transpose(1, 2, 0)
                        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                        image_np = image_np.astype(np.uint8)

                        # Run inference
                        boxes, labels, scores = self.predict(image_np)

                        # Store predictions and targets for mAP calculation
                        val_predictions.append({
                            'boxes': boxes,
                            'labels': labels,
                            'scores': scores
                        })

                        val_targets.append({
                            'boxes': target['boxes'].numpy(),
                            'labels': target['labels'].numpy() - 1  # -1 to match the predicted labels
                        })

                # Calculate mAP
                mAP = self._calculate_map_from_predictions(val_predictions, val_targets)
                history['val_map'].append(mAP)

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Cls Loss: {epoch_cls_loss:.4f}, "
                      f"Reg Loss: {epoch_reg_loss:.4f}, "
                      f"Val mAP: {mAP:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
            else:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, "
                      f"Cls Loss: {epoch_cls_loss:.4f}, "
                      f"Reg Loss: {epoch_reg_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

        return history

    def _calculate_map_from_predictions(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate mAP from predictions and targets

        Args:
            predictions (list): List of dictionaries with 'boxes', 'labels', and 'scores'
            targets (list): List of dictionaries with 'boxes' and 'labels'
            iou_threshold (float): IoU threshold for mAP calculation

        Returns:
            float: mAP value
        """
        # Convert predictions and targets to format expected by _calculate_map
        all_pred_boxes = [p['boxes'] for p in predictions]
        all_pred_scores = [p['scores'] for p in predictions]
        all_pred_labels = [p['labels'] for p in predictions]
        all_gt_boxes = [t['boxes'] for t in targets]
        all_gt_labels = [t['labels'] for t in targets]

        return self._calculate_map(all_pred_boxes, all_pred_scores, all_pred_labels,
                                  all_gt_boxes, all_gt_labels, iou_threshold)
        """
        Run inference on an image

        Args:
            image: Image to run inference on (numpy array or PIL image)
            confidence_threshold (float): Confidence threshold for detections
            nms_threshold (float): NMS threshold
            max_proposals (int): Maximum number of proposals to process

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
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Image not found: {image}")

        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy
            if image.dim() == 3:  # [C, H, W]
                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img_np = img_np.astype(np.uint8)
            else:
                raise ValueError(f"Invalid image tensor shape: {image.shape}")
        elif isinstance(image, np.ndarray):
            # Already numpy
            img_np = image
        elif isinstance(image, Image.Image):
            # Convert PIL to numpy
            img_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Generate region proposals
        proposals = self._generate_proposals(img_np, max_proposals)

        # Normalize image for feature extraction
        transform = self._get_transform(train=False)
        transformed = transform(image=img_np, bboxes=[], labels=[])
        img_tensor = transformed['image']

        # Extract features from proposals
        features = self._extract_features(img_tensor, proposals)

        if len(features) == 0:
            return np.array([]), np.array([]), np.array([])

        # Classify proposals
        with torch.no_grad():
            class_scores = self.model['classifier'](features)
            class_probs = F.softmax(class_scores, dim=1)

            # Get class predictions
            max_scores, pred_labels = torch.max(class_probs[:, 1:], dim=1)  # Skip background class (0)
            pred_labels = pred_labels + 1  # Add 1 because we skipped background

            # Get bounding box refinement
            bbox_deltas = self.model['bbox_regressor'](features)

        # Convert to numpy
        proposals = proposals[:len(features)]  # Match proposals to features
        scores = max_scores.cpu().numpy()
        labels = pred_labels.cpu().numpy()
        deltas = bbox_deltas.cpu().numpy()

        # Apply bounding box regression
        boxes = []
        for i, (proposal, label, delta) in enumerate(zip(proposals, labels, deltas)):
            # Get the deltas for this class
            idx = (label - 1) * 4  # -1 because label includes background
            tx, ty, tw, th = delta[idx:idx+4]

            # Get original box coordinates
            x1, y1, x2, y2 = proposal
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2
            cy = y1 + h/2

            # Apply deltas
            cx = cx + tx * w
            cy = cy + ty * h
            w = w * np.exp(tw)
            h = h * np.exp(th)

            # Convert back to [x1, y1, x2, y2] format
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            boxes.append([x1, y1, x2, y2])

        boxes = np.array(boxes)

        # Filter by confidence threshold
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Apply non-maximum suppression
        if len(boxes) > 0:
            # Perform NMS for each class separately
            final_boxes = []
            final_scores = []
            final_labels = []

            for cls in range(1, self.num_classes + 1):
                cls_mask = labels == cls
                if not np.any(cls_mask):
                    continue

                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]

                # Convert to torch tensors for NMS
                cls_boxes_tensor = torch.from_numpy(cls_boxes).float()
                cls_scores_tensor = torch.from_numpy(cls_scores).float()

                # Apply NMS
                keep_idx = nms(cls_boxes_tensor, cls_scores_tensor, nms_threshold)

                # Convert keep indices back to numpy
                keep = keep_idx.cpu().numpy()

                # Add to final lists
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_labels.append(np.full(len(keep), cls))

            # Combine results from all classes
            if final_boxes:
                boxes = np.vstack(final_boxes)
                scores = np.hstack(final_scores)
                labels = np.hstack(final_labels)
            else:
                boxes = np.array([])
                scores = np.array([])
                labels = np.array([])

        # Convert class indices back to original indices (undo the +1 that was added for background)
        labels = labels - 1

        return boxes, labels, scores