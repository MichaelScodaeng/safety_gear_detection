"""
Implementation of Fast R-CNN algorithm for Safety Gear Detection.
"""

import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms, RoIPool
from torchvision.transforms import functional as TF
from PIL import Image

from .base import RCNNBase


class FastRCNN(RCNNBase):
    """
    Implementation of Fast R-CNN
    Fast R-CNN improves on R-CNN by:
    1. Sharing the convolutional features across all proposals
    2. Using RoI pooling to extract fixed-size feature maps for each proposal
    3. Training the classifier and bounding box regressor in a single stage
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize Fast R-CNN

        Args:
            num_classes (int): Number of classes to detect
            device (str): Device to use (cuda or cpu)
            config (dict): Configuration parameters
        """
        super().__init__(num_classes, device, config)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Fast R-CNN model with a pre-trained backbone"""
        # Load a pre-trained ResNet model for feature extraction
        backbone = torchvision.models.resnet50(weights="DEFAULT")

        # Remove the last fully connected layer and pooling layer
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Create RoI pooling layer
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1 / 16)

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Create class score predictor
        self.class_predictor = nn.Linear(4096, self.num_classes + 1)  # +1 for background class

        # Create bounding box regressor
        self.bbox_regressor = nn.Linear(4096, 4 * self.num_classes)  # (tx, ty, tw, th) for each class

        # Combine into a single model
        self.model = nn.ModuleDict({
            'backbone': self.backbone,
            'classifier': self.classifier,
            'class_predictor': self.class_predictor,
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

    def _extract_roi_features(self, features, proposals, image_shape):
        """
        Extract ROI features using ROI pooling

        Args:
            features (torch.Tensor): Feature map from backbone [B, C, H, W]
            proposals (np.ndarray): Region proposals in [x1, y1, x2, y2] format
            image_shape (tuple): Original image shape (H, W)

        Returns:
            torch.Tensor: ROI features
        """
        # Convert proposals to torch tensor
        rois = torch.from_numpy(proposals).float().to(self.device)

        # Add batch dimension (all from same image)
        batch_indices = torch.zeros(rois.shape[0], 1, device=self.device)
        rois_with_batch = torch.cat([batch_indices, rois], dim=1)

        # Normalize ROIs to feature map scale
        h_ratio = features.shape[2] / image_shape[0]
        w_ratio = features.shape[3] / image_shape[1]
        rois_with_batch[:, 1] *= w_ratio
        rois_with_batch[:, 2] *= h_ratio
        rois_with_batch[:, 3] *= w_ratio
        rois_with_batch[:, 4] *= h_ratio

        # Apply ROI pooling
        roi_features = self.roi_pool(features, rois_with_batch)

        return roi_features

    def predict(self, image, confidence_threshold=0.5, nms_threshold=0.3, max_proposals=300):
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

        # Store original image shape
        original_shape = img_np.shape[:2]

        # Generate region proposals
        proposals = self._generate_proposals(img_np, max_proposals)

        if len(proposals) == 0:
            return np.array([]), np.array([]), np.array([])

        # Normalize image for feature extraction
        transform = self._get_transform(train=False)
        transformed = transform(image=img_np, bboxes=[], labels=[])
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Extract features from the entire image
        with torch.no_grad():
            features = self.model['backbone'](img_tensor)

            # Extract ROI features
            roi_features = self._extract_roi_features(features, proposals, original_shape)

            # Classify ROIs
            roi_pooled = self.model['classifier'](roi_features)
            class_scores = self.model['class_predictor'](roi_pooled)
            class_probs = F.softmax(class_scores, dim=1)

            # Get class predictions
            max_scores, pred_labels = torch.max(class_probs[:, 1:], dim=1)  # Skip background class (0)
            pred_labels = pred_labels + 1  # Add 1 because we skipped background

            # Get bounding box refinement
            bbox_deltas = self.model['bbox_regressor'](roi_pooled)

        # Convert to numpy
        scores = max_scores.cpu().numpy()
        labels = pred_labels.cpu().numpy()
        deltas = bbox_deltas.cpu().numpy()

        # Apply bounding box regression
        boxes = []
        final_labels = []
        final_scores = []

        for i, (proposal, label, score) in enumerate(zip(proposals, labels, scores)):
            # Skip background detections
            if label == 0 or score < confidence_threshold:
                continue

            # Get the deltas for this class
            idx = (label - 1) * 4  # -1 because label includes background
            tx, ty, tw, th = deltas[i, idx:idx + 4]

            # Get original box coordinates
            x1, y1, x2, y2 = proposal
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            # Apply deltas
            cx = cx + tx * w
            cy = cy + ty * h
            w = w * np.exp(tw)
            h = h * np.exp(th)

            # Convert back to [x1, y1, x2, y2] format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_shape[1], x2)
            y2 = min(original_shape[0], y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            final_labels.append(label)
            final_scores.append(score)

        if not boxes:
            return np.array([]), np.array([]), np.array([])

        boxes = np.array(boxes)
        final_labels = np.array(final_labels)
        final_scores = np.array(final_scores)

        # Apply non-maximum suppression
        if len(boxes) > 0:
            # Perform NMS for each class separately
            final_boxes = []
            final_scores_filtered = []
            final_labels_filtered = []

            for cls in range(1, self.num_classes + 1):
                cls_mask = final_labels == cls
                if not np.any(cls_mask):
                    continue

                cls_boxes = boxes[cls_mask]
                cls_scores = final_scores[cls_mask]

                # Convert to torch tensors for NMS
                cls_boxes_tensor = torch.from_numpy(cls_boxes).float()
                cls_scores_tensor = torch.from_numpy(cls_scores).float()

                # Apply NMS
                keep_idx = nms(cls_boxes_tensor, cls_scores_tensor, nms_threshold)

                # Convert keep indices back to numpy
                keep = keep_idx.cpu().numpy()

                # Add to final lists
                final_boxes.append(cls_boxes[keep])
                final_scores_filtered.append(cls_scores[keep])
                final_labels_filtered.append(np.full(len(keep), cls))

            # Combine results from all classes
            if final_boxes:
                boxes = np.vstack(final_boxes)
                scores = np.hstack(final_scores_filtered)
                labels = np.hstack(final_labels_filtered)
            else:
                boxes = np.array([])
                scores = np.array([])
                labels = np.array([])

        # Convert class indices back to original indices (undo the +1 that was added for background)
        labels = labels - 1

        return boxes, labels, scores

    def train(self, train_dataset, valid_dataset=None, epochs=10, lr=0.001, weight_decay=0.0005, batch_size=4,
              pos_iou_threshold=0.5, neg_iou_threshold=0.3, proposals_per_image=128, max_proposals=300):
        """
        Train the Fast R-CNN model

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
            num_batches = 0

            # Track time
            start_time = time.time()

            # Iterate through dataset
            for idx in range(0, len(train_dataset), batch_size):
                batch_images = []
                batch_proposals = []
                batch_proposal_labels = []
                batch_proposal_boxes = []
                batch_img_shapes = []

                # Process each image in the batch
                for batch_idx in range(idx, min(idx + batch_size, len(train_dataset))):
                    # Get data
                    image, target = train_dataset[batch_idx]

                    # Convert tensor to numpy for proposal generation
                    if isinstance(image, torch.Tensor):
                        image_np = image.numpy().transpose(1, 2, 0)
                        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                        image_np = image_np.astype(np.uint8)
                    else:
                        image_np = image

                    # Store image shape
                    img_shape = image_np.shape[:2]
                    batch_img_shapes.append(img_shape)

                    # Generate proposals
                    proposals = self._generate_proposals(image_np, max_proposals=max_proposals)

                    # Skip if no proposals
                    if len(proposals) == 0:
                        continue

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
                    proposal_reg_targets = np.zeros((len(proposals), 4 * self.num_classes), dtype=np.float32)

                    # Assign positive samples
                    pos_indices = np.where(max_ious >= pos_iou_threshold)[0]
                    for i in pos_indices:
                        gt_idx = gt_argmax[i]
                        cls_idx = gt_labels[gt_idx] - 1  # -1 because we add 1 during dataset creation
                        proposal_labels[i] = gt_labels[gt_idx]

                        # Calculate regression targets
                        proposal_box = proposals[i]
                        gt_box = gt_boxes[gt_idx]

                        # Convert to center + size format
                        p_cx = (proposal_box[0] + proposal_box[2]) / 2
                        p_cy = (proposal_box[1] + proposal_box[3]) / 2
                        p_w = proposal_box[2] - proposal_box[0]
                        p_h = proposal_box[3] - proposal_box[1]

                        g_cx = (gt_box[0] + gt_box[2]) / 2
                        g_cy = (gt_box[1] + gt_box[3]) / 2
                        g_w = gt_box[2] - gt_box[0]
                        g_h = gt_box[3] - gt_box[1]

                        # Calculate regression targets
                        tx = (g_cx - p_cx) / p_w
                        ty = (g_cy - p_cy) / p_h
                        tw = np.log(g_w / p_w)
                        th = np.log(g_h / p_h)

                        # Store regression targets for this class
                        reg_idx = cls_idx * 4
                        proposal_reg_targets[i, reg_idx:reg_idx + 4] = [tx, ty, tw, th]

                    # Assign negative samples
                    neg_indices = np.where((max_ious < pos_iou_threshold) & (max_ious >= neg_iou_threshold))[0]
                    proposal_labels[neg_indices] = 0  # Background class

                    # Limit number of proposals for training (balanced sampling)
                    pos_count = min(len(pos_indices), proposals_per_image // 2)
                    if len(pos_indices) > pos_count:
                        pos_indices = np.random.choice(pos_indices, pos_count, replace=False)

                    neg_count = min(len(neg_indices), proposals_per_image - pos_count)
                    if len(neg_indices) > neg_count:
                        neg_indices = np.random.choice(neg_indices, neg_count, replace=False)

                    # Combine positive and negative indices
                    keep_indices = np.concatenate([pos_indices, neg_indices])

                    # Store batch data
                    batch_images.append(image)
                    batch_proposals.append(proposals[keep_indices])
                    batch_proposal_labels.append(proposal_labels[keep_indices])
                    batch_proposal_boxes.append(proposal_reg_targets[keep_indices])

                # Skip if no valid images in batch
                if not batch_images:
                    continue

                # Convert images to tensor batch
                image_tensors = torch.stack([img.to(self.device) if isinstance(img, torch.Tensor)
                                             else torch.from_numpy(img).to(self.device) for img in batch_images])

                # Extract features for all images in batch
                features = self.model['backbone'](image_tensors)

                # Process each image's proposals
                all_roi_features = []
                all_proposal_labels = []
                all_proposal_reg_targets = []

                for i, (img_proposals, img_labels, img_reg_targets, img_shape) in enumerate(
                        zip(batch_proposals, batch_proposal_labels, batch_proposal_boxes, batch_img_shapes)):
                    # Extract ROI features
                    roi_features = self._extract_roi_features(features[i:i + 1], img_proposals, img_shape)

                    # Add to batch data
                    all_roi_features.append(roi_features)
                    all_proposal_labels.extend(img_labels)
                    all_proposal_reg_targets.append(img_reg_targets)

                # Skip if no valid ROIs
                if not all_roi_features:
                    continue

                # Concatenate ROI features
                roi_features_cat = torch.cat(all_roi_features, dim=0)
                proposal_labels_tensor = torch.tensor(all_proposal_labels, device=self.device)
                proposal_reg_targets_tensor = torch.tensor(np.concatenate(all_proposal_reg_targets), device=self.device)

                # Forward pass
                roi_pooled = self.model['classifier'](roi_features_cat)
                class_scores = self.model['class_predictor'](roi_pooled)
                bbox_pred = self.model['bbox_regressor'](roi_pooled)

                # Compute classification loss
                cls_loss = classification_loss_fn(class_scores, proposal_labels_tensor)

                # Compute regression loss for positive samples
                pos_mask = proposal_labels_tensor > 0
                if torch.any(pos_mask):
                    # Get positive class indices
                    pos_labels = proposal_labels_tensor[pos_mask] - 1  # -1 because background is 0

                    # Select predictions for correct classes
                    reg_pred = torch.zeros((pos_mask.sum(), 4), device=self.device)
                    for i, (idx, cls_idx) in enumerate(zip(torch.where(pos_mask)[0], pos_labels)):
                        reg_idx = cls_idx * 4
                        reg_pred[i] = bbox_pred[idx, reg_idx:reg_idx + 4]

                    # Get regression targets for positive samples
                    reg_targets = torch.zeros((pos_mask.sum(), 4), device=self.device)
                    for i, (idx, cls_idx) in enumerate(zip(torch.where(pos_mask)[0], pos_labels)):
                        reg_idx = cls_idx * 4
                        reg_targets[i] = proposal_reg_targets_tensor[idx, reg_idx:reg_idx + 4]

                    # Compute regression loss
                    reg_loss = regression_loss_fn(reg_pred, reg_targets)
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
                num_batches += 1

                # Print progress
                if (num_batches) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {num_batches}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Cls Loss: {cls_loss.item():.4f}, "
                          f"Reg Loss: {reg_loss.item():.4f}")

            # Update learning rate
            lr_scheduler.step()

            # Calculate average loss
            if num_batches > 0:
                epoch_loss /= num_batches
                epoch_cls_loss /= num_batches
                epoch_reg_loss /= num_batches
                history['train_loss'].append(epoch_loss)

                # Validation
                if valid_dataset:
                    # Set model to evaluation mode
                    self.model.eval()

                    # Calculate mAP on validation set
                    val_predictions = []
                    val_targets = []

                    with torch.no_grad():
                        for idx in range(len(valid_dataset)):
                            # Get data
                            image, target = valid_dataset[idx]

                            # Run inference
                            boxes, labels, scores = self.predict(image)

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

                    # Print results
                    print(f"Epoch {epoch + 1}/{epochs}, "
                          f"Train Loss: {epoch_loss:.4f}, "
                          f"Cls Loss: {epoch_cls_loss:.4f}, "
                          f"Reg Loss: {epoch_reg_loss:.4f}, "
                          f"Val mAP: {mAP:.4f}, "
                          f"Time: {time.time() - start_time:.2f}s")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, "
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