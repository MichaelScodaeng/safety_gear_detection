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
        print(f"Unfreezing layers: {unfreeze_layers}")
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
    
    if use_amp:
        print("Using mixed precision training")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_map': []
    }
    
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
        for images, targets in progress_bar:
            accumulated_steps += 1
            
            '''#Test
            if accumulated_steps == 20:
                break'''
            
            try:
                # Move to device
                images = [img.to(detector.device) for img in images]
                targets = [{k: v.to(detector.device) for k, v in t.items()} for t in targets]
                
                # Zero gradients for first step or after update
                if accumulated_steps == 1 or accumulated_steps % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Forward pass with automatic mixed precision
                if use_amp:
                    with autocast():
                        loss_dict = detector.model.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        
                        # Scale loss based on accumulation steps
                        scaled_loss = losses / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(scaled_loss).backward()
                    
                    # Update if reached accumulation steps
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    # Standard forward pass
                    loss_dict = detector.model.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Scale loss based on accumulation steps
                    scaled_loss = losses / gradient_accumulation_steps
                    
                    # Backward pass
                    scaled_loss.backward()
                    
                    # Update if reached accumulation steps
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        optimizer.step()
                
                # Update metrics
                epoch_loss += losses.item()
                
                # Update component losses
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
        
        # Gradually unfreeze deeper layers as training progresses (uncomment if needed)
        '''
        if epoch == int(epochs * 0.3):  # After 30% of epochs
            print("Unfreezing layer4 of the backbone")
            for name, param in detector.model.model.named_parameters():
                if "backbone.layer4" in name:
                    param.requires_grad = True
        
        elif epoch == int(epochs * 0.6):  # After 60% of epochs
            print("Unfreezing FPN layers")
            for name, param in detector.model.model.named_parameters():
                if "fpn" in name:
                    param.requires_grad = True
        '''
        
        # Save model checkpoint
        detector.save_model(f"{detector.config.get('OUTPUT_PATH', './')}/model_epoch_{epoch+1}.pt")
    
    return history