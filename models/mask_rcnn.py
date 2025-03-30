"""
Implementation of Mask R-CNN algorithm for Safety Gear Detection.
"""

import os
import time
import numpy as np
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, 
    maskrcnn_resnet50_fpn_v2,
)
# Add this import for the classifier head replacement
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image

from .base import RCNNBase

from config import CFG

class MaskRCNN_Model(RCNNBase):
    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize Mask R-CNN model.

        Args:
            num_classes (int): Number of classes (including background).
            device (str): Device to use for computation.
            config (dict): Configuration parameters.
        """
        super().__init__(num_classes, device, config)
        self.num_classes = num_classes
        self.model = None
        self.device = device if device else CFG.DEVICE
        self.config = config.copy() if config else CFG.__dict__.copy()
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Mask R-CNN model with a ResNet backbone and FPN.
        """
        # Load a pre-trained Mask R-CNN model
        model_type = self.config.get('model_type')
        if model_type == 'maskrcnn_resnet50_fpn':
            self.model = maskrcnn_resnet50_fpn(pretrained=True)
        elif model_type == 'maskrcnn_resnet50_fpn_v2':
            self.model = maskrcnn_resnet50_fpn_v2(pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        print(f"Model type: {model_type}")
        
        # Replace the classifier with a new one, that has num_classes (including background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        self.model.roi_heads.box_predictor = box_predictor

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes + 1
        )
        
        # Print model info
        self.print_hyperparameters()
        
        # Move model to device
        self.model.to(self.device)


            # Move model to the specified device
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
            print("MASK R-CNN MODEL CONFIGURATION")
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