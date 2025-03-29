"""
Dataset implementation for the Safety Gear Detection System.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SafetyGearDataset(Dataset):
    """Dataset class for safety gear detection"""

    def __init__(self, img_dir, label_dir, transform=None, class_map=None):
        """
        Initialize the dataset

        Args:
            img_dir (str): Directory containing images
            label_dir (str): Directory containing labels
            transform: Transforms to apply to images
            class_map (dict): Mapping from class IDs to class names
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Get all image files
        self.img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.endswith(('.jpg', '.jpeg', '.png'))])

        # Class mapping
        self.class_map = class_map if class_map else {}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #print(f"Loading image: {img_path}")
        #print(f"Image shape: {image.shape}")

        # Get label file path
        img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")
        #print(f"Looking for label file: {label_path}")

        boxes = []
        labels = []

        if os.path.exists(label_path):
            # YOLO format: class_id, x_center, y_center, width, height (normalized)
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id, x_center, y_center, width, height = map(float, data)

                        # Convert normalized YOLO format to pixel coordinates [x1, y1, x2, y2]
                        img_h, img_w = image.shape[:2]
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h

                        # Ensure coordinates are within image boundaries
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue

                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32) if len(transformed['bboxes']) > 0 else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64) if len(transformed['labels']) > 0 else np.zeros((0,), dtype=np.int64)
            #print(f"Type of transformed['bboxes']: {type(transformed['bboxes'])}")
            #print(f"Value of transformed['bboxes']: {transformed['bboxes']}")

        # Create target dictionary for torchvision detection models
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels + 1, dtype=torch.int64),  # Add 1 because 0 is background in torchvision models
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target

def get_transforms(train=False):
    """
    Get transforms for data preprocessing
    
    Args:
        train (bool): Whether to use training transforms
        
    Returns:
        A.Compose: Composition of transforms
    """
    if train:
        return A.Compose([
            #A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
def collate_fn(batch):
        return tuple(zip(*batch))
def create_data_loaders(train_dir, valid_dir=None, test_dir=None, batch_size=4,shuffle = True):
    """
    Create data loaders for training, validation and testing
    
    Args:
        train_dir (str): Directory containing training data
        valid_dir (str, optional): Directory containing validation data
        test_dir (str, optional): Directory containing test data
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Define transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create datasets
    train_dataset = None
    if train_dir:
        train_img_dir = os.path.join(train_dir, 'images')
        train_label_dir = os.path.join(train_dir, 'labels')
        if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
            train_dataset = SafetyGearDataset(train_img_dir, train_label_dir, transform=train_transform)
    
    valid_dataset = None
    if valid_dir:
        valid_img_dir = os.path.join(valid_dir, 'images')
        valid_label_dir = os.path.join(valid_dir, 'labels')
        if os.path.exists(valid_img_dir) and os.path.exists(valid_label_dir):
            valid_dataset = SafetyGearDataset(valid_img_dir, valid_label_dir, transform=val_transform)
    
    test_dataset = None
    if test_dir:
        test_img_dir = os.path.join(test_dir, 'images')
        test_label_dir = os.path.join(test_dir, 'labels')
        if os.path.exists(test_img_dir) and os.path.exists(test_label_dir):
            test_dataset = SafetyGearDataset(test_img_dir, test_label_dir, transform=val_transform)
    
    # Custom collate function for data loaders
    
    
    # Create data loaders
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    return train_loader, valid_loader, test_loader
