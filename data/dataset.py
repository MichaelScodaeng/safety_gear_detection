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
from transformers import DetrImageProcessor
import pycocotools.coco as coco
import json

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
            # Convert tensor labels to list of integers
            if isinstance(target["labels"], torch.Tensor):
                labels_list = target["labels"].tolist()
            else:
                labels_list = target["labels"]
                
            # Convert tensor boxes to list if needed
            if isinstance(target["boxes"], torch.Tensor):
                boxes_list = target["boxes"].tolist()
            else:
                boxes_list = target["boxes"]
                
            # Apply transform with native Python types
            transformed = self.transform(image=np.array(image), bboxes=boxes_list, labels=labels_list)
            
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

class DETRDataset(torch.utils.data.Dataset):
    """
    Dataset for DETR (Detection Transformer) models.
    This dataset works with COCO-format data.
    """
    def __init__(self, img_dir, ann_file, processor, transform=None):
        """
        Initialize DETR dataset.
        
        Args:
            img_dir (str): Path to images directory
            ann_file (str): Path to COCO annotation file
            processor (DetrImageProcessor): DETR image processor for preprocessing
            transform: Optional transforms
        """
        from pycocotools.coco import COCO
        
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.processor = processor
        self.transform = transform
        
        # Get image IDs
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
        # Get category mapping to ensure proper class indexing
        self.cat_mapping = {}
        for i, cat_id in enumerate(sorted(self.coco.getCatIds())):
            self.cat_mapping[cat_id] = i  # Map original category IDs to 0-indexed sequential IDs
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (pixel_values, target)
        """
        from PIL import Image
        import numpy as np
        import torch
        import os
        
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_file = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_file).convert('RGB')
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        area = []
        iscrowd = []
        
        for ann in anns:
            # Extract bounding box
            x, y, width, height = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + width, y + height])
            # Map to sequential 0-indexed labels
            cat_id = ann['category_id']
            labels.append(self.cat_mapping[cat_id])  # Use our mapping to ensure 0-indexed sequential classes
            # Get or calculate area
            area.append(ann.get('area', width * height))
            # Get iscrowd flag
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Prepare target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']])
        }
        
        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        
        # Process image with DETR processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        return pixel_values, target

def create_detr_data_loaders(data_dir=None, batch_size=4, train_dir=None, valid_dir=None, test_dir=None):
    """
    Create data loaders for DETR models using a COCO-formatted dataset.
    
    Args:
        data_dir (str): Base path to the DETR dataset
        batch_size (int): Batch size
        train_dir (str, optional): Specific path to training data
        valid_dir (str, optional): Specific path to validation data
        test_dir (str, optional): Specific path to test data
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    import os
    import torch
    from transformers import DetrImageProcessor
    
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'detr-data')
    
    print(f"Using DETR data from: {data_dir}")
    
    # Check if the data directory exists
    if not os.path.exists(data_dir) and not (train_dir or valid_dir or test_dir):
        print(f"DETR data directory not found: {data_dir}")
        return None, None, None
    
    # Initialize image processor from Hugging Face
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Important: Set proper configs to handle data with 0-indexed classes
    # This avoids the "weight tensor should be defined either for all or no classes" error
    processor.format = "coco_detection"
    processor.max_boxes = 100  # Adjust based on your dataset
    
    train_loader, valid_loader, test_loader = None, None, None
    
    # Process train data
    if train_dir:
        # Use directly provided train_dir
        print(f"Using provided training directory: {train_dir}")
        train_ann_file_options = [
            os.path.join(train_dir, '_annotations.coco.json'),
            os.path.join(os.path.dirname(train_dir), 'annotations', 'instances_train.json'),
            os.path.join(train_dir, 'annotations', '_annotations.coco.json')
        ]
        
        train_ann_file = None
        for ann_option in train_ann_file_options:
            if os.path.exists(ann_option):
                train_ann_file = ann_option
                print(f"Using annotations from {train_ann_file}")
                break
                
        if train_ann_file and os.path.exists(train_ann_file):
            try:
                train_dataset = DETRDataset(train_dir, train_ann_file, processor)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True, 
                    collate_fn=detr_collate_fn
                )
                print(f"Created training dataloader with {len(train_dataset)} samples")
            except Exception as e:
                print(f"Error creating training dataloader: {e}")
        else:
            print(f"Warning: Could not find valid annotation file for training data")
    else:
        # Find the correct directory structure
        # First try the expected structure: data_dir/train
        train_dir_options = [
            os.path.join(data_dir, 'train'),
            os.path.join(data_dir, 'images', 'train'),
            data_dir  # If images are directly in data_dir
        ]
        
        train_dir = None
        train_ann_file = None
        
        # Find the first valid train directory and annotation file
        for dir_option in train_dir_options:
            if os.path.exists(dir_option):
                # Try different annotation file locations
                ann_file_options = [
                    os.path.join(dir_option, '_annotations.coco.json'),
                    os.path.join(data_dir, 'annotations', 'instances_train.json'),
                    os.path.join(dir_option, 'annotations', '_annotations.coco.json')
                ]
                
                for ann_option in ann_file_options:
                    if os.path.exists(ann_option):
                        train_dir = dir_option
                        train_ann_file = ann_option
                        print(f"Found training data at {train_dir}")
                        print(f"Using annotations from {train_ann_file}")
                        break
                
                if train_dir:
                    break
        
        # Create training data loader if files were found
        if train_dir and train_ann_file and os.path.exists(train_ann_file):
            try:
                train_dataset = DETRDataset(train_dir, train_ann_file, processor)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True, 
                    collate_fn=collate_fn_detr
                )
                print(f"Created training dataloader with {len(train_dataset)} samples")
            except Exception as e:
                print(f"Error creating training dataloader: {e}")
        else:
            print(f"Warning: Could not find valid training data and annotations")
    
    # Process validation data
    if valid_dir:
        # Use directly provided valid_dir
        print(f"Using provided validation directory: {valid_dir}")
        valid_ann_file_options = [
            os.path.join(valid_dir, '_annotations.coco.json'),
            os.path.join(os.path.dirname(valid_dir), 'annotations', 'instances_val.json'),
            os.path.join(valid_dir, 'annotations', '_annotations.coco.json')
        ]
        
        valid_ann_file = None
        for ann_option in valid_ann_file_options:
            if os.path.exists(ann_option):
                valid_ann_file = ann_option
                print(f"Using annotations from {valid_ann_file}")
                break
                
        if valid_ann_file and os.path.exists(valid_ann_file):
            try:
                valid_dataset = DETRDataset(valid_dir, valid_ann_file, processor)
                valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=batch_size,
                    shuffle=False, 
                    collate_fn=collate_fn_detr
                )
                print(f"Created validation dataloader with {len(valid_dataset)} samples")
            except Exception as e:
                print(f"Error creating validation dataloader: {e}")
        else:
            print(f"Warning: Could not find valid annotation file for validation data")
    else:
        # Similar approach for validation data
        valid_dir_options = [
            os.path.join(data_dir, 'valid'),
            os.path.join(data_dir, 'val'),
            os.path.join(data_dir, 'images', 'val'),
            os.path.join(data_dir, 'images', 'valid')
        ]
        
        valid_dir = None
        valid_ann_file = None
        
        for dir_option in valid_dir_options:
            if os.path.exists(dir_option):
                ann_file_options = [
                    os.path.join(dir_option, '_annotations.coco.json'),
                    os.path.join(data_dir, 'annotations', 'instances_val.json'),
                    os.path.join(dir_option, 'annotations', '_annotations.coco.json')
                ]
                
                for ann_option in ann_file_options:
                    if os.path.exists(ann_option):
                        valid_dir = dir_option
                        valid_ann_file = ann_option
                        print(f"Found validation data at {valid_dir}")
                        print(f"Using annotations from {valid_ann_file}")
                        break
                
                if valid_dir:
                    break
        
        if valid_dir and valid_ann_file and os.path.exists(valid_ann_file):
            try:
                valid_dataset = DETRDataset(valid_dir, valid_ann_file, processor)
                valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=batch_size,
                    shuffle=False, 
                    collate_fn=collate_fn_detr
                )
                print(f"Created validation dataloader with {len(valid_dataset)} samples")
            except Exception as e:
                print(f"Error creating validation dataloader: {e}")
    
    # Process test data
    if test_dir:
        # Use directly provided test_dir
        print(f"Using provided test directory: {test_dir}")
        test_ann_file_options = [
            os.path.join(test_dir, '_annotations.coco.json'),
            os.path.join(os.path.dirname(test_dir), 'annotations', 'instances_test.json'),
            os.path.join(test_dir, 'annotations', '_annotations.coco.json')
        ]
        
        test_ann_file = None
        for ann_option in test_ann_file_options:
            if os.path.exists(ann_option):
                test_ann_file = ann_option
                print(f"Using annotations from {test_ann_file}")
                break
                
        if test_ann_file and os.path.exists(test_ann_file):
            try:
                test_dataset = DETRDataset(test_dir, test_ann_file, processor)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False, 
                    collate_fn=detr_collate_fn
                )
                print(f"Created test dataloader with {len(test_dataset)} samples")
            except Exception as e:
                print(f"Error creating test dataloader: {e}")
        else:
            print(f"Warning: Could not find valid annotation file for test data")
    else:
        # Similar approach for test data
        test_dir_options = [
            os.path.join(data_dir, 'test'),
            os.path.join(data_dir, 'images', 'test')
        ]
        
        test_dir = None
        test_ann_file = None
        
        for dir_option in test_dir_options:
            if os.path.exists(dir_option):
                ann_file_options = [
                    os.path.join(dir_option, '_annotations.coco.json'),
                    os.path.join(data_dir, 'annotations', 'instances_test.json'),
                    os.path.join(dir_option, 'annotations', '_annotations.coco.json')
                ]
                
                for ann_option in ann_file_options:
                    if os.path.exists(ann_option):
                        test_dir = dir_option
                        test_ann_file = ann_option
                        print(f"Found test data at {test_dir}")
                        print(f"Using annotations from {test_ann_file}")
                        break
                
                if test_dir:
                    break
        
        if test_dir and test_ann_file and os.path.exists(test_ann_file):
            try:
                test_dataset = DETRDataset(test_dir, test_ann_file, processor)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False, 
                    collate_fn=detr_collate_fn
                )
                print(f"Created test dataloader with {len(test_dataset)} samples")
            except Exception as e:
                print(f"Error creating test dataloader: {e}")
    
    return train_loader, valid_loader, test_loader

def detr_collate_fn(batch):
    """
    Custom collate function for DETR data loader.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        tuple: (images, targets)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images
    images = torch.stack(images)
    
    return images, targets

