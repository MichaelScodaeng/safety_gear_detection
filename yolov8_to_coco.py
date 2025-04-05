#!/usr/bin/env python3
import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path

def yolo_to_coco(yolo_dataset_path, output_json_path, image_width=None, image_height=None):
    """
    Convert YOLOv8 format dataset to COCO format.
    
    Args:
        yolo_dataset_path: Path to the YOLOv8 dataset (containing images and labels folders)
        output_json_path: Path to save the COCO JSON output
        image_width: Optional fixed width for all images (if not provided, will read from images)
        image_height: Optional fixed height for all images (if not provided, will read from images)
    """
    # Initialize COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get class names from classes.txt if available
    classes_file = os.path.join(yolo_dataset_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_list = [line.strip() for line in f.readlines()]
    else:
        # Default class list (you should modify this to match your specific dataset)
        print("Warning: classes.txt not found. Using default class list.")
        class_list = ["class_0", "class_1", "class_2"]  # Placeholder class list
    
    # Create categories in COCO format
    for i, class_name in enumerate(class_list):
        coco_format["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })
    
    # Paths setup
    images_dir = os.path.join(yolo_dataset_path, "images")
    labels_dir = os.path.join(yolo_dataset_path, "labels")
    
    # If specific train/val/test directories exist, handle them
    if not os.path.exists(images_dir):
        images_dir = os.path.join(yolo_dataset_path, "train", "images")
        if not os.path.exists(images_dir):
            raise ValueError(f"Could not find images directory at {yolo_dataset_path}/images or {yolo_dataset_path}/train/images")
    
    if not os.path.exists(labels_dir):
        labels_dir = os.path.join(yolo_dataset_path, "train", "labels")
        if not os.path.exists(labels_dir):
            raise ValueError(f"Could not find labels directory at {yolo_dataset_path}/labels or {yolo_dataset_path}/train/labels")
    
    # Get image files
    image_files = sorted(glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True) + 
                         glob.glob(os.path.join(images_dir, "**", "*.jpeg"), recursive=True) +
                         glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    annotation_id = 0
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image and its annotation
    for image_id, image_file in enumerate(tqdm(image_files)):
        # Get image filename (without extension) to match with label file
        image_filename = os.path.basename(image_file)
        image_stem = os.path.splitext(image_filename)[0]
        
        # Determine image dimensions
        if image_width is None or image_height is None:
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening {image_file}: {e}")
                continue
        else:
            width, height = image_width, image_height
        
        # Add image info to COCO format
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })
        
        # Construct label file path (YOLOv8 format: one .txt file per image with same name)
        label_file = os.path.join(labels_dir, f"{image_stem}.txt")
        
        # If label file doesn't exist, continue to next image
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {image_filename}")
            continue
        
        # Process annotations from the label file
        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                # Parse YOLOv8 format: class_id x_center y_center width height
                # All values are normalized to [0, 1]
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Skipping invalid annotation in {label_file}: {line}")
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                
                # Convert normalized coordinates to absolute pixel values
                x_min = int((x_center - box_width / 2) * width)
                y_min = int((y_center - box_height / 2) * height)
                box_width_px = int(box_width * width)
                box_height_px = int(box_height * height)
                
                # Ensure coordinates are within image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                box_width_px = min(width - x_min, box_width_px)
                box_height_px = min(height - y_min, box_height_px)
                
                # Skip invalid boxes
                if box_width_px <= 0 or box_height_px <= 0:
                    print(f"Warning: Skipping invalid box in {label_file}: {line}")
                    continue
                
                # Add annotation to COCO format
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width_px, box_height_px],
                    "area": box_width_px * box_height_px,
                    "segmentation": [],  # No segmentation in basic YOLOv8
                    "iscrowd": 0
                })
                
                annotation_id += 1
    
    # Write output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Conversion completed. Output saved to {output_json_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Total categories: {len(coco_format['categories'])}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv8 format dataset to COCO format')
    parser.add_argument('--yolo-path', required=True, help='Path to YOLOv8 dataset directory')
    parser.add_argument('--output', required=True, help='Path to save the COCO JSON output')
    parser.add_argument('--width', type=int, help='Fixed width for all images (optional)')
    parser.add_argument('--height', type=int, help='Fixed height for all images (optional)')
    parser.add_argument('--split', default=None, choices=['train', 'val', 'test'], 
                        help='Dataset split to convert (optional)')
    
    args = parser.parse_args()
    
    yolo_path = args.yolo_path
    output_path = args.output
    
    # Handle specific dataset split
    if args.split:
        yolo_path = os.path.join(yolo_path, args.split)
        if not os.path.exists(yolo_path):
            raise ValueError(f"Split directory not found: {yolo_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    yolo_to_coco(
        yolo_dataset_path=yolo_path,
        output_json_path=output_path,
        image_width=args.width,
        image_height=args.height
    )


if __name__ == "__main__":
    main()