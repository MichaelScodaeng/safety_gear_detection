"""
Training utilities for the Safety Gear Detection System.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import traceback
import contextlib

def print_model_info(detector, training_args=None):
    """
    Print detailed information about model hyperparameters and training settings
    
    Args:
        detector: SafetyGearDetector instance
        training_args (dict, optional): Training arguments including learning rate, 
                                    batch size, epochs, etc.
    """
    # Import for pretty printing
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
        
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    
    # Model architecture details
    model_info = [
        ["Model Type", detector.model_type],
        ["Number of Classes", f"{detector.num_classes} (+ 1 background)"],
        ["Device", detector.device],
    ]
    
    # Get class names if available
    class_names = detector.config.get('CLASS_NAMES', [])
    if class_names:
        model_info.append(["Classes", ", ".join(class_names)])
    
    # Model parameters
    total_params = sum(p.numel() for p in detector.model.model.parameters())
    trainable_params = sum(p.numel() for p in detector.model.model.parameters() if p.requires_grad)
    model_info.extend([
        ["Total Parameters", f"{total_params:,}"],
        ["Trainable Parameters", f"{trainable_params:,} ({trainable_params/total_params:.2%})"],
        ["Frozen Parameters", f"{total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})"]
    ])
    
    # Model size in MB
    model_size_mb = sum(p.numel() * p.element_size() for p in detector.model.model.parameters()) / (1024 * 1024)
    model_info.append(["Model Size", f"{model_size_mb:.2f} MB"])
    
    # Extract model-specific parameters if available
    try:
        if hasattr(detector.model.model, "rpn"):
            # RPN Parameters
            anchor_sizes = detector.model.model.rpn.anchor_generator.sizes
            anchor_ratios = detector.model.model.rpn.anchor_generator.aspect_ratios
            rpn_fg_iou_thresh = getattr(detector.model.model.rpn, 'fg_iou_thresh', 'N/A')
            rpn_bg_iou_thresh = getattr(detector.model.model.rpn, 'bg_iou_thresh', 'N/A')
            
            model_info.extend([
                ["RPN Anchor Sizes", str(anchor_sizes)],
                ["RPN Anchor Ratios", str(anchor_ratios)],
                ["RPN FG IoU Threshold", str(rpn_fg_iou_thresh)],
                ["RPN BG IoU Threshold", str(rpn_bg_iou_thresh)]
            ])
        
        if hasattr(detector.model.model, "roi_heads"):
            # ROI Parameters
            box_score_thresh = detector.model.model.roi_heads.score_thresh
            box_nms_thresh = detector.model.model.roi_heads.nms_thresh
            box_detections_per_img = detector.model.model.roi_heads.detections_per_img
            box_fg_iou_thresh = getattr(detector.model.model.roi_heads, 'fg_iou_thresh', 'N/A')
            box_bg_iou_thresh = getattr(detector.model.model.roi_heads, 'bg_iou_thresh', 'N/A')
            
            model_info.extend([
                ["Box Score Threshold", str(box_score_thresh)],
                ["Box NMS Threshold", str(box_nms_thresh)],
                ["Max Detections per Image", str(box_detections_per_img)],
                ["Box FG IoU Threshold", str(box_fg_iou_thresh)],
                ["Box BG IoU Threshold", str(box_bg_iou_thresh)]
            ])
        
        if hasattr(detector.model.model, "transform"):
            # Image transform parameters
            min_size = detector.model.model.transform.min_size[0] if isinstance(detector.model.model.transform.min_size, tuple) else detector.model.model.transform.min_size
            max_size = detector.model.model.transform.max_size
            
            model_info.extend([
                ["Min Image Size", str(min_size)],
                ["Max Image Size", str(max_size)]
            ])
    except Exception as e:
        print(f"Could not extract all model parameters: {e}")
    
    # Print model architecture details
    if use_tabulate:
        print(tabulate(model_info, headers=["Parameter", "Value"], tablefmt="grid"))
    else:
        for param, value in model_info:
            print(f"{param:30} {value}")
    print(detector.model.model)
    for name, param in detector.model.model.named_parameters():
        print(f"Layer: {name}, requires_grad: {param.requires_grad}")
def validate_labels(detector, train_loader):
    """
    Validate that all labels in the dataset are defined in the CLASS_NAMES configuration.
    
    Args:
        detector: SafetyGearDetector instance
        train_loader: DataLoader for training data
    """
    class_names = detector.config.get('CLASS_NAMES', [])
    if not class_names:
        raise ValueError("CLASS_NAMES configuration is empty or not defined.")
    
    valid_labels = set(range(len(class_names)))
    print(f"Valid labels: {valid_labels}")
    
    # For DETR models, skip detailed validation as it's handled by the DETR processor
    if detector.model_type.startswith('detr'):
        print("DETR model detected - skipping detailed dataset validation")
        return
    
    for batch in train_loader:
        _, targets = batch
        for target in targets:
            # Convert tensor to list if necessary
            if isinstance(target['labels'], torch.Tensor):
                labels = target['labels'].tolist()
            else:
                labels = target['labels']
                
            for label in labels:
                if label not in valid_labels:
                    raise ValueError(f"Undefined label {label} found in dataset. ")

def train_model(detector, train_loader, valid_loader=None, epochs=10, 
                lr=0.001, weight_decay=0.0005, batch_size=4, 
                fine_tune=False, freeze_backbone=True, unfreeze_layers=None,
                gradient_accumulation_steps=1):
    """
    Train the model
    
    Args:
        detector: SafetyGearDetector instance
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of epochs to train
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        batch_size: Batch size
        fine_tune: Whether to fine-tune the model
        freeze_backbone: Whether to freeze the backbone
        unfreeze_layers: Layers to unfreeze if freeze_backbone is True
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        dict: Training history
    """
    print(f"Training {detector.model_type} model for {epochs} epochs")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}, Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Create optimizer
    params = [p for p in detector.model.model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    
    
    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Freeze backbone if specified
    if freeze_backbone:
        print("Freezing backbone")
        for name, param in detector.model.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
    
    # Unfreeze specific layers if specified
    if unfreeze_layers is not None:
        print(f"Currently Unfreezing layers: {unfreeze_layers}")
        for name, param in detector.model.model.named_parameters():
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
    
    # Initialize mixed precision training if available
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print("Currently using GPUs/TPUs:", torch.cuda.device_count())
    print("GPU/TPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Using device:", detector.device)
    print("Unfreezed layers:", unfreeze_layers if unfreeze_layers else "None")
    print("Unfreezed backbone:", freeze_backbone)
    print("Fine-tuning:", fine_tune)
    
    if use_amp:
        print("Using mixed precision training")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_map': []
    }
    
    # Prepare training arguments for printing
    training_args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'fine_tune': fine_tune,
        'freeze_backbone': freeze_backbone,
        'unfreeze_layers': unfreeze_layers,
        'use_amp': torch.cuda.is_available(),
        'optimizer': optimizer
    }
    
    # Print model information
    print_model_info(detector, training_args)

    # Start training
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        detector.model.model.train()
        
        # Training metrics
        epoch_loss = 0.0
        epoch_loss_classifier = 0.0
        epoch_loss_box_reg = 0.0
        epoch_loss_objectness = 0.0
        epoch_loss_rpn_box_reg = 0.0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Track gradient accumulation
        accumulated_steps = 0
        
        # Training loop
        if detector.model_type.startswith('detr'):
            # DETR requires tensors, not lists of images
           for batch in progress_bar:
            images, targets = batch  # Unpack the tuple
            accumulated_steps += 1
            
            try:
                # Move images to device and stack them if they're individual tensors
                if isinstance(images, list):
                    # Process list of images
                    pixel_values = torch.stack([img.to(detector.device) for img in images])
                else:
                    # Already stacked tensor
                    pixel_values = images.to(detector.device)
                
                # Format targets for DETR
                detr_targets = []
                for idx, target in enumerate(targets):
                    # Ensure labels are tensors
                    if not isinstance(target['labels'], torch.Tensor):
                        labels = torch.tensor(target['labels'], device=detector.device)
                    else:
                        labels = target['labels'].to(detector.device)
                    
                    # Ensure boxes are tensors
                    if not isinstance(target['boxes'], torch.Tensor):
                        boxes = torch.tensor(target['boxes'], device=detector.device)
                    else:
                        boxes = target['boxes'].to(detector.device)
                    
                    # Create DETR format target
                    detr_target = {
                        'labels': labels,
                        'boxes': boxes,
                        'image_id': torch.tensor([idx], device=detector.device)
                    }
                    
                    # Calculate area if not present
                    if 'area' not in target:
                        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        detr_target['area'] = area
                    else:
                        area_tensor = target['area'].to(detector.device) if isinstance(target['area'], torch.Tensor) else torch.tensor(target['area'], device=detector.device)
                        detr_target['area'] = area_tensor
                    
                    detr_targets.append(detr_target)
                
                # Zero gradients
                if accumulated_steps == 1 or accumulated_steps % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Forward pass for DETR
                outputs = detector.model.model(pixel_values=pixel_values, labels=detr_targets)
                losses = outputs.loss
                loss_dict = outputs.loss_dict
                
                # Scale loss for gradient accumulation
                scaled_loss = losses / gradient_accumulation_steps
                
                # Backward pass
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    scaled_loss.backward()
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        optimizer.step()
                
                # Update metrics
                epoch_loss += losses.item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=losses.item())
                
            except Exception as e:
                print(f"Error in batch: {e}")
                traceback.print_exc()
                optimizer.zero_grad()
                continue
        elif detector.model_type.startswith('fasterrcnn'):
            # Standard training for Faster R-CNN and other models
            for batch in progress_bar:
                images, targets = batch
                accumulated_steps += 1
                
                try:
                    # Convert any non-tensor data to tensors if needed
                    if not isinstance(images[0], torch.Tensor):
                        # Convert image list to tensor format if needed
                        # This depends on your specific implementation
                        pass
                    
                    # Make sure targets are properly formatted
                    for t in targets:
                        # Convert labels to native Python integers for albumentations
                        if isinstance(t['labels'], torch.Tensor):
                            t['labels'] = t['labels'].tolist()
                        if isinstance(t['boxes'], torch.Tensor):
                            t['boxes'] = t['boxes'].tolist()
                    
                    # Move to device
                    images = [img.to(detector.device) for img in images]
                    targets = [{k: v.to(detector.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=detector.device) 
                              for k, v in t.items()} for t in targets]
                    
                    # Zero gradients
                    if accumulated_steps == 1 or accumulated_steps % gradient_accumulation_steps == 0:
                        optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext():
                        loss_dict = detector.model.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = losses / gradient_accumulation_steps
                    
                    # Backward pass
                    if use_amp:
                        scaler.scale(scaled_loss).backward()
                        if accumulated_steps % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        scaled_loss.backward()
                        if accumulated_steps % gradient_accumulation_steps == 0:
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
                    progress_bar.set_postfix(loss=losses.item())
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    traceback.print_exc()
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
        if valid_loader:
            # Use validate method for validation
            val_loss, val_map = detector.validate(valid_loader)
            
            history['val_loss'].append(val_loss)
            history['val_map'].append(val_map)
            print(f"Val Loss  mAP@0.5 : {-val_loss:.4f}, mAP@0.5:0.05:0.95 : {val_map:.4f}")
            
            # Update learning rate
            lr_scheduler.step(val_loss)
        
        # Gradually unfreeze deeper layers as training progresses
        if epoch == 0:  # First epoch - train only the classifier head
            for name, param in detector.model.model.named_parameters():
                if "roi_heads.box_predictor" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif epoch == int(epochs * 0.2):  # After 20% of epochs
            print("Unfreezing layer4 of the backbone")
            for name, param in detector.model.model.named_parameters():
                if "backbone.layer4" in name or "roi_heads" in name:
                    param.requires_grad = True

        elif epoch == int(epochs * 0.4):  # After 40% of epochs
            print("Unfreezing layer3 and FPN")
            for name, param in detector.model.model.named_parameters():
                if "backbone.layer3" in name or "fpn" in name:
                    param.requires_grad = True
        
        # Save model checkpoint
        detector.save_model(f"{detector.config.get('OUTPUT_PATH', './')}/model_epoch_{epoch+1}.pt")
    
    return history