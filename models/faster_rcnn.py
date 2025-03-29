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
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn, 
    fasterrcnn_mobilenet_v3_large_320_fpn
)
# Add this import for the classifier head replacement
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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

    def print_model_info(self, model):
        """Print model parameters and structure information"""
        # Count and print total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Print model architecture summary
        print("\nModel Architecture:")
        print(model)

    def _initialize_model(self):
        """Initialize the Faster R-CNN model with a pre-trained backbone"""
        # Get model type from config or use custom implementation
        model_type = self.config.get('model_type', 'custom')
        
        if model_type == 'custom':
            print("Using custom implementation of Faster R-CNN")
            # Using custom torchvision implementation
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
                min_size=640,
                max_size=640,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,
                box_detections_per_img=100
            )
        else:
            from torchvision.ops import MultiScaleRoIAlign
            # Using pre-trained torchvision models   
            # Use a more flexible ROI pooler to handle various feature map sizes
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # Use all FPN levels
                output_size=7,                       # Standard output size
                sampling_ratio=2                     # Use bilinear sampling
            )
    
            print(f"Using pre-trained model: {model_type}")
                    # Use pre-built torchvision models
                    # First step: Initialize with pre-trained weights (91 classes for COCO)
            if model_type == 'fasterrcnn_resnet50_fpn':
                self.model = fasterrcnn_resnet50_fpn(
                    weights="DEFAULT",  # Use pre-trained weights (91 classes)
                    box_score_thresh=0.05,
                    box_nms_thresh=0.5,
                    box_detections_per_img=100,
                    min_size=640,
                    max_size=640,
                    #box_roi_pool=roi_pooler
                )
            elif model_type == 'fasterrcnn_resnet50_fpn_v2':
                self.model = fasterrcnn_resnet50_fpn_v2(
                    weights="DEFAULT",  # Use pre-trained weights (91 classes)
                    box_score_thresh=0.05,
                    box_nms_thresh=0.5,
                    box_detections_per_img=100,
                    min_size=640,
                    max_size=640,
                    #box_roi_pool=roi_pooler
                )
            elif model_type == 'fasterrcnn_mobilenet_v3_large_fpn':
                self.model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights="DEFAULT",  # Use pre-trained weights (91 classes)
                    box_score_thresh=0.05,
                    box_nms_thresh=0.5,
                    box_detections_per_img=100,
                    min_size=640,
                    max_size=640,
                    #box_roi_pool=roi_pooler
                )
            elif model_type == 'fasterrcnn_mobilenet_v3_large_320_fpn':
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights="DEFAULT",  # Use pre-trained weights (91 classes)
                    box_score_thresh=0.05,
                    box_nms_thresh=0.5,
                    box_detections_per_img=100,
                    min_size=640,
                    max_size=640,
                    #box_roi_pool=roi_pooler
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}. " 
                                 f"Choose from: 'custom', 'fasterrcnn_resnet50_fpn', "
                                 f"'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_fpn', "
                                 f"'fasterrcnn_mobilenet_v3_large_320_fpn'")
            
            # Second step: Replace the box predictor head with a new one for our classes
            if model_type != 'custom':
                # Get the number of input features for the classifier
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                
                # Replace the pre-trained head with a new one for our number of classes
                self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
                
                print(f"Replaced classification head with a new one for {self.num_classes} classes + background")
        
        # Print model info
        self.print_model_info(self.model)
        
        # Move model to device
        self.model.to(self.device)

    def fine_tune(self, model_path=None, freeze_backbone=True, unfreeze_layers=None):
        """
        Fine-tune the model with granular control over which layers to unfreeze

        Args:
            model_path (str, optional): Path to pre-trained model
            freeze_backbone (bool): Whether to freeze the backbone
            unfreeze_layers (list, optional): List of layer names to unfreeze (even if backbone is frozen)
                                            Examples: ['layer4', 'fpn', 'rpn']
        """
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.load_model(model_path)

        # Start by freezing all parameters if requested
        if freeze_backbone:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model type from config
        model_type = self.config.get('model_type', 'custom')
        
        # Selectively unfreeze layers based on the model type
        if unfreeze_layers:
            for name, param in self.model.named_parameters():
                # Check if any specified layer is in the parameter name
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
                    print(f"Unfreezing: {name}")
        
        # Always unfreeze the classifier head (box predictor)
        # This is critical since we replaced it with a new one
        for param in self.model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
            
        print("Box predictor layers are always trainable")
        
        # Print summary of trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})")
    def print_hyperparameters(self, training_args=None):
        """
        Print detailed information about model hyperparameters and training settings
        
        Args:
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
        print("FASTER R-CNN MODEL CONFIGURATION")
        print("="*80)
        
        # Model architecture details
        model_info = [
            ["Model Type", self.config.get('model_type', 'custom')],
            ["Number of Classes", f"{self.num_classes} (+ 1 background)"],
            ["Device", self.device],
        ]
        
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_info.extend([
            ["Total Parameters", f"{total_params:,}"],
            ["Trainable Parameters", f"{trainable_params:,} ({trainable_params/total_params:.2%})"],
            ["Frozen Parameters", f"{total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})"]
        ])
        
        # Model size in MB
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        model_info.append(["Model Size", f"{model_size_mb:.2f} MB"])
        
        # RPN Parameters
        try:
            anchor_sizes = self.model.rpn.anchor_generator.sizes
            anchor_ratios = self.model.rpn.anchor_generator.aspect_ratios
            rpn_fg_iou_thresh = getattr(self.model.rpn, 'fg_iou_thresh', 'N/A')
            rpn_bg_iou_thresh = getattr(self.model.rpn, 'bg_iou_thresh', 'N/A')
            
            model_info.extend([
                ["RPN Anchor Sizes", str(anchor_sizes)],
                ["RPN Anchor Ratios", str(anchor_ratios)],
                ["RPN FG IoU Threshold", str(rpn_fg_iou_thresh)],
                ["RPN BG IoU Threshold", str(rpn_bg_iou_thresh)]
            ])
        except Exception as e:
            print(f"Could not extract RPN parameters: {e}")
        
        # ROI Parameters
        try:
            box_score_thresh = self.model.roi_heads.score_thresh
            box_nms_thresh = self.model.roi_heads.nms_thresh
            box_detections_per_img = self.model.roi_heads.detections_per_img
            box_fg_iou_thresh = getattr(self.model.roi_heads, 'fg_iou_thresh', 'N/A')
            box_bg_iou_thresh = getattr(self.model.roi_heads, 'bg_iou_thresh', 'N/A')
            
            model_info.extend([
                ["Box Score Threshold", str(box_score_thresh)],
                ["Box NMS Threshold", str(box_nms_thresh)],
                ["Max Detections per Image", str(box_detections_per_img)],
                ["Box FG IoU Threshold", str(box_fg_iou_thresh)],
                ["Box BG IoU Threshold", str(box_bg_iou_thresh)]
            ])
        except Exception as e:
            print(f"Could not extract ROI parameters: {e}")
        
        # Image transform parameters
        try:
            min_size = self.model.transform.min_size[0] if isinstance(self.model.transform.min_size, tuple) else self.model.transform.min_size
            max_size = self.model.transform.max_size
            
            model_info.extend([
                ["Min Image Size", str(min_size)],
                ["Max Image Size", str(max_size)]
            ])
        except Exception as e:
            print(f"Could not extract transform parameters: {e}")
        
        # Print model architecture details
        if use_tabulate:
            print(tabulate(model_info, headers=["Parameter", "Value"], tablefmt="grid"))
        else:
            for param, value in model_info:
                print(f"{param:30} {value}")
        
        # Print training parameters if provided
        if training_args:
            print("\n" + "="*80)
            print("TRAINING CONFIGURATION")
            print("="*80)
            
            train_info = [
                ["Epochs", training_args.get('epochs', 'N/A')],
                ["Batch Size", training_args.get('batch_size', 'N/A')],
                ["Learning Rate", training_args.get('lr', 'N/A')],
                ["Weight Decay", training_args.get('weight_decay', 'N/A')],
                ["Gradient Accumulation Steps", training_args.get('gradient_accumulation_steps', 'N/A')],
                ["Fine-tuning", training_args.get('fine_tune', 'N/A')],
                ["Freeze Backbone", training_args.get('freeze_backbone', 'N/A')],
                ["Unfreeze Layers", training_args.get('unfreeze_layers', 'N/A')],
                ["Mixed Precision", training_args.get('use_amp', 'N/A')]
            ]
            
            # Print training parameters
            if use_tabulate:
                print(tabulate(train_info, headers=["Parameter", "Value"], tablefmt="grid"))
            else:
                for param, value in train_info:
                    print(f"{param:30} {value}")
            
            # Print optimizer details if provided
            if 'optimizer' in training_args:
                opt = training_args['optimizer']
                opt_type = type(opt).__name__
                print("\nOptimizer:", opt_type)
                
                # Print learning rates for each parameter group
                if hasattr(opt, 'param_groups'):
                    print("\nParameter Groups:")
                    for i, group in enumerate(opt.param_groups):
                        group_info = [
                            ["Group", i],
                            ["Learning Rate", group.get('lr', 'N/A')],
                            ["Weight Decay", group.get('weight_decay', 'N/A')],
                            ["Parameters", len(group.get('params', []))]
                        ]
                        
                        if use_tabulate:
                            print(tabulate(group_info, headers=["Property", "Value"], tablefmt="simple"))
                        else:
                            for prop, val in group_info:
                                print(f"  {prop}: {val}")
                        print()
        
        # Print CUDA memory stats if available
        if torch.cuda.is_available():
            print("\n" + "="*80)
            print("CUDA MEMORY USAGE")
            print("="*80)
            
            cuda_info = [
                ["CUDA Device", torch.cuda.get_device_name(0)],
                ["Memory Allocated", f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MB"],
                ["Memory Reserved", f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB"],
                ["Max Memory Allocated", f"{torch.cuda.max_memory_allocated(0)/1024**2:.2f} MB"]
            ]
            
            if use_tabulate:
                print(tabulate(cuda_info, headers=["Metric", "Value"], tablefmt="grid"))
            else:
                for metric, value in cuda_info:
                    print(f"{metric:30} {value}")
        
        print("\n" + "="*80 + "\n")