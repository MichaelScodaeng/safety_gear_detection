"""
Dataset implementation for the Safety Gear Detection System.
"""
import yaml 
from config import CFG
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

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

    def _parse_yolo_annotation(self, label_path, image_shape):
        """
        Parse YOLOv8 annotation file and convert to absolute pixel values.

        Args:
            label_path (str): Path to the YOLO annotation file.
            image_shape (tuple): Shape of the image (height, width).

        Returns:
            dict: Parsed annotations with 'boxes' and 'labels'.
        """
        h, w = image_shape
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_min = (x_center - width / 2) * w
                    y_min = (y_center - height / 2) * h
                    x_max = (x_center + width / 2) * w
                    y_max = (y_center + height / 2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))

        return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = os.path.join(self.label_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_shape = image.size[::-1]  # (height, width)

        # Parse annotations
        target = self._parse_yolo_annotation(label_path, image_shape)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=target["boxes"], labels=target["labels"])
            image = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)

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

def collate_fn_detr(batch):
    """
    Custom collate function for DETR models that stacks images into a tensor batch
    and keeps targets as a list.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

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
"""
YOLO-specific dataset implementation for the Safety Gear Detection System.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class YOLODataset(Dataset):
    """Dataset class for YOLO-based safety gear detection"""

    def __init__(self, img_dir, label_dir, img_size=640, transform=None, class_map=None):
        """
        Initialize the dataset

        Args:
            img_dir (str): Directory containing images
            label_dir (str): Directory containing labels
            img_size (int): Image size for YOLO inference
            transform: Additional transforms to apply (beyond resizing)
            class_map (dict): Mapping from class IDs to class names
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size

        # Get all image files
        self.img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.endswith(('.jpg', '.jpeg', '.png'))])

        # Class mapping
        self.class_map = class_map if class_map else {}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Get image and label paths
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")
        
        # For Ultralytics, we can simply return the paths
        # This is the most efficient way to use YOLO with Ultralytics
        return img_path, label_path
    
    # Add this to your dataset.py or create a new file

def create_yolo_data_loaders(train_dir, valid_dir, test_dir, batch_size=4, img_size=640):
    """Create YOLO-specific data loaders for training, validation, and testing."""
    # Create data.yaml file for YOLO training with ABSOLUTE paths
    data_yaml = {
        'train': os.path.abspath(os.path.join(train_dir, 'images')),
        'val': os.path.abspath(os.path.join(valid_dir, 'images')),
        'test': os.path.abspath(os.path.join(test_dir, 'images')),
        'nc': len(CFG.CLASS_NAMES),
        'names': CFG.CLASS_NAMES
    }
    
    # Print the paths to verify they're correct
    print(f"YOLO Dataset Paths:")
    print(f"  Train: {data_yaml['train']}")
    print(f"  Val: {data_yaml['val']}")
    print(f"  Test: {data_yaml['test']}")
    
    # Save data.yaml in current directory
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    # For YOLO models, we don't actually need to create DataLoader objects
    # The YOLO framework handles data loading internally
    return None, None, None

class DETRDataset(SafetyGearDataset):
    """Dataset class specifically for DETR models"""
    
    def __getitem__(self, idx):
        """Get an item from the dataset, converting tensors to numpy arrays for albumentations"""
        try:
            image, target = super().__getitem__(idx)
            
            # Convert tensor labels to numpy arrays for albumentations compatibility
            if isinstance(target['labels'], torch.Tensor):
                target['labels'] = target['labels'].numpy()
            
            # Convert tensor boxes to numpy arrays if needed
            if isinstance(target['boxes'], torch.Tensor):
                target['boxes'] = target['boxes'].numpy()
                
            return image, target
        except Exception as e:
            print(f"Error in DETRDataset.__getitem__ for index {idx}: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_detr_data_loaders(train_dir, valid_dir, test_dir, batch_size=4, shuffle=True):
    """Create data loaders specifically for DETR models"""
    
    # Use the specialized DETR dataset class
    train_dataset = DETRDataset(train_dir, get_transforms(train=True)) if train_dir and os.path.exists(train_dir) else None
    valid_dataset = DETRDataset(valid_dir, get_transforms(train=False)) if valid_dir and os.path.exists(valid_dir) else None
    test_dataset = DETRDataset(test_dir, get_transforms(train=False)) if test_dir and os.path.exists(test_dir) else None
    
    # Use the standard collate function since we're still using the base dataset format
    collate_fn = lambda batch: tuple(zip(*batch))
    
    # Create data loaders
    # Use DETRDataset if it exists, otherwise fallback to SafetyGearDataset with special handling
    try:
        print("Attempting to use DETRDataset...")
        train_dataset = DETRDataset(train_dir, get_transforms(train=True)) if train_dir and os.path.exists(train_dir) else None
        valid_dataset = DETRDataset(valid_dir, get_transforms(train=False)) if valid_dir and os.path.exists(valid_dir) else None
        test_dataset = DETRDataset(test_dir, get_transforms(train=False)) if test_dir and os.path.exists(test_dir) else None
    except NameError:
        print("DETRDataset not found, falling back to SafetyGearDataset...")
        # If DETRDataset is not defined, use SafetyGearDataset with special collate function
        train_dataset = SafetyGearDataset(train_dir, get_transforms(train=True)) if train_dir and os.path.exists(train_dir) else None
        valid_dataset = SafetyGearDataset(valid_dir, get_transforms(train=False)) if valid_dir and os.path.exists(valid_dir) else None
        test_dataset = SafetyGearDataset(test_dir, get_transforms(train=False)) if test_dir and os.path.exists(test_dir) else None
    
    # Debug dataset sizes
    print(f"- Train dataset size: {len(train_dataset) if train_dataset else 0}")
    print(f"- Valid dataset size: {len(valid_dataset) if valid_dataset else 0}")
    print(f"- Test dataset size: {len(test_dataset) if test_dataset else 0}")
    
    # Custom collate function for DETR
    def collate_fn_detr(batch):
        try:
            # Process labels to convert tensors to lists/numpy to avoid albumentations issues
            processed_batch = []
            for img, target in batch:
                if isinstance(target.get('labels'), torch.Tensor):
                    target['labels'] = target['labels'].tolist()  # Convert to list to avoid tensor issues
                if isinstance(target.get('boxes'), torch.Tensor):
                    target['boxes'] = target['boxes'].numpy().tolist()  # Convert to list to avoid tensor issues
                processed_batch.append((img, target))
            
            # Standard collate function
            return tuple(zip(*processed_batch))
        except Exception as e:
            print(f"Error in collate_fn_detr: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Create data loaders with error handling
    try:
        print("Creating train loader...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn_detr,
            num_workers=0
        ) if train_dataset else None
        
        print("Creating validation loader...")
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_detr,
            num_workers=0
        ) if valid_dataset else None
        
        print("Creating test loader...")
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_detr,
            num_workers=0
        ) if test_dataset else None
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return train_loader, valid_loader, test_loader