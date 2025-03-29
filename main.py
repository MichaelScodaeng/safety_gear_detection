"""
Main script for the Safety Gear Detection System.
"""

import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from config import CFG
from data.dataset import SafetyGearDataset, get_transforms, create_data_loaders
from detector import SafetyGearDetector


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Safety Gear Detection System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Operation mode (train, evaluate, predict)')
    parser.add_argument('--model_type', type=str, default='fasterrcnn_resnet50_fpn_v2',
                        choices=['custom', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 
                                'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn'],
                        help='Model architecture to use')
    parser.add_argument('--data_path', type=str, default=CFG.CSS_DATA_PATH,
                        help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the saved model (for evaluation or prediction)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image for prediction')
    parser.add_argument('--epochs', type=int, default=CFG.EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=CFG.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune an existing model')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone during fine-tuning')
    parser.add_argument('--unfreeze_layers', type=str, nargs='+', default=None,
                        help='Layers to unfreeze during fine-tuning (e.g., --unfreeze_layers layer4 fpn rpn)')
    parser.add_argument('--confidence_threshold', type=float, default=CFG.CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for predictions')
    parser.add_argument('--nms_threshold', type=float, default=CFG.NMS_THRESHOLD,
                        help='NMS threshold for predictions')
    parser.add_argument('--output_dir', type=str, default=CFG.OUTPUT_PATH,
                        help='Directory to save outputs')
    return parser.parse_args()

def visualize_dataloader_sample(dataloader, class_names):
    """
    Visualize a sample from the dataloader alongside its original version.
    """
    if dataloader is None:
        print("Dataloader is empty. Cannot visualize samples.")
        return
    
    # Get the dataset first
    dataset = dataloader.dataset
    # Choose a specific index to visualize
    sample_idx = 0
    
    # Get the original image and label directly from the dataset
    img_path = dataset.img_files[sample_idx]
    original_image = plt.imread(img_path)
    
    # Get annotations from label file
    img_name = os.path.basename(img_path).rsplit('.', 1)[0]
    label_path = os.path.join(dataset.label_dir, f"{img_name}.txt")
    
    # Parse the original annotations
    boxes = []
    labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) == 5:
                    class_id, x_center, y_center, width, height = map(float, data)
                    
                    # Convert normalized YOLO format to pixel coordinates
                    img_h, img_w = original_image.shape[:2]
                    x1 = (x_center - width/2) * img_w
                    y1 = (y_center - height/2) * img_h
                    x2 = (x_center + width/2) * img_w
                    y2 = (y_center + height/2) * img_h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id))
    
    original_boxes = np.array(boxes)
    original_labels = np.array(labels)
    
    # Get the transformed version of the SAME image directly from the dataset
    # This ensures we compare the same image before and after transformation
    transformed_image, transformed_target = dataset[sample_idx]
    transformed_image = transformed_image.permute(1, 2, 0).numpy()
    
    # Denormalize the image for better visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transformed_image = std * transformed_image + mean
    transformed_image = np.clip(transformed_image, 0, 1)
    
    transformed_boxes = transformed_target['boxes'].numpy()
    transformed_labels = transformed_target['labels'].numpy() - 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original image
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    for box, label in zip(original_boxes, original_labels):
        x1, y1, x2, y2 = box
        ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    edgecolor='red', facecolor='none', linewidth=2))
        if class_names and label < len(class_names):
            ax1.text(x1, y1 - 5, class_names[label], color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.5))
    ax1.axis('off')
    
    # Plot transformed image
    ax2.imshow(transformed_image)
    ax2.set_title("Transformed Image")
    for box, label in zip(transformed_boxes, transformed_labels):
        x1, y1, x2, y2 = box
        ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    edgecolor='red', facecolor='none', linewidth=2))
        if class_names and label < len(class_names):
            ax2.text(x1, y1 - 5, class_names[label], color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.5))
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_model(args):
    """Train a new model or fine-tune an existing one."""
    print(f"Training {args.model_type} model...")
    print("Torch Available: ", torch.cuda.is_available())
    
    # Create data loaders
    train_dir = os.path.join(args.data_path, 'train')
    valid_dir = os.path.join(args.data_path, 'val')
    test_dir = os.path.join(args.data_path, 'test')

    print("Training Path: ", train_dir)
    print("Validation Path: ", valid_dir)
    print("Test Path: ", test_dir)
    
    # Create data loaders directly
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dir, valid_dir, test_dir, batch_size=args.batch_size, shuffle=True
    )

    if train_loader is None or valid_loader is None or test_loader is None:
        print("No training or validation or test data found. Please check the data path.")
        return None
    
    # Create the detector with the specified model type
    detector = SafetyGearDetector(model_type=args.model_type)
    
    # If fine-tuning, apply fine-tuning settings
    if args.fine_tune and args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path} for fine-tuning")
        detector.load_model(args.model_path)
        detector.model.fine_tune(
            freeze_backbone=args.freeze_backbone,
            unfreeze_layers=args.unfreeze_layers
        )
    
    # Train the model
    history = detector.train(
        train_loader,
        valid_loader, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY,
        fine_tune=args.fine_tune,
        freeze_backbone=args.freeze_backbone,
        unfreeze_layers=args.unfreeze_layers
    )
    
    # Save the model
    model_save_path = os.path.join(args.output_dir, f"{args.model_type}_safety_gear.pt")
    detector.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return detector


def evaluate_model(args):
    """Evaluate a model on the test set."""
    print(f"Evaluating {args.model_type} model...")
    
    # Create detector
    detector = SafetyGearDetector(model_type=args.model_type)
    
    # Load model
    if args.model_path and os.path.exists(args.model_path):
        detector.load_model(args.model_path)
    else:
        default_path = f"{args.output_dir}/{args.model_type}_safety_gear.pt"
        if os.path.exists(default_path):
            detector.load_model(default_path)
        else:
            print(f"No model found at {args.model_path or default_path}. Training a new model...")
            detector = train_model(args)
            if detector is None:
                return
    
    # Create test data loader
    test_dir = os.path.join(args.data_path, 'test')
    _, _, test_loader = create_data_loaders(None, None, test_dir, batch_size=args.batch_size)
    
    if test_loader is None:
        print("No test data found. Please check the data path.")
        return
    
    # Evaluate the model
    metrics = detector.evaluate(test_loader)
    print(f"mAP: {metrics['mAP']:.4f}")
    
    return metrics


def predict_image(args):
    """Run inference on an image."""
    print(f"Running inference with {args.model_type} model...")
    
    # Create detector
    detector = SafetyGearDetector(model_type=args.model_type)
    
    # Load model
    if args.model_path and os.path.exists(args.model_path):
        detector.load_model(args.model_path)
    else:
        default_path = f"{args.output_dir}/{args.model_type}_safety_gear.pt"
        if os.path.exists(default_path):
            detector.load_model(default_path)
        else:
            print(f"No model found at {args.model_path or default_path}. Training a new model...")
            detector = train_model(args)
            if detector is None:
                return
    
    # Get image path
    image_path = args.image_path
    if image_path is None or not os.path.exists(image_path):
        # Find a test image
        test_dir = os.path.join(args.data_path, 'test', 'images')
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                image_path = os.path.join(test_dir, images[0])
                print(f"No image path provided. Using test image: {image_path}")
            else:
                print("No test images found. Please provide an image path.")
                return
        else:
            print("No test directory found. Please provide an image path.")
            return
    
    # Run inference
    boxes, labels, scores, class_names = detector.predict(
        image_path,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # Print results
    print(f"Found {len(boxes)} objects:")
    for i, (box, label, score, class_name) in enumerate(zip(boxes, labels, scores, class_names)):
        x1, y1, x2, y2 = box
        print(f"  {i+1}. {class_name}: {score:.4f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Visualize results
    result_image = detector.visualize_prediction(image_path, boxes, labels, scores)
    
    # Save output image
    output_path = f"{args.output_dir}/detection_result.jpg"
    plt.imsave(output_path, result_image)
    print(f"Result saved to {output_path}")
    
    return boxes, labels, scores, class_names


def visualize_dataloader_sample(dataloader, class_names):
    """
    Visualize a sample from the dataloader alongside its original version.
    
    Args:
        dataloader: DataLoader containing batches of images and targets
        class_names: List of class names for visualization
    """
    if dataloader is None:
        print("Dataloader is empty. Cannot visualize samples.")
        return
    
    # Get a batch of data
    for images, targets in dataloader:
        # Get the original image
        dataset = dataloader.dataset
        idx = 0  # Use the first image in the dataset
        
        # Get original image path
        img_path = dataset.img_files[idx]  # Use img_files instead of img_dir
        
        # Get label file path for annotations
        img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(dataset.label_dir, f"{img_name}.txt")
        
        # Load original image without transforms
        original_image = plt.imread(img_path)
        
        # Parse the original annotations
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id, x_center, y_center, width, height = map(float, data)
                        
                        # Convert normalized YOLO format to pixel coordinates
                        img_h, img_w = original_image.shape[:2]
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))
        
        original_boxes = np.array(boxes)
        original_labels = np.array(labels)
        
        # Get the transformed image from the dataloader
        transformed_image = images[0].permute(1, 2, 0).numpy()  # Convert to HWC format
        transformed_boxes = targets[0]['boxes'].numpy()
        transformed_labels = targets[0]['labels'].numpy() - 1  # Subtract 1 to convert back from torchvision format
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot original image
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        for box, label in zip(original_boxes, original_labels):
            x1, y1, x2, y2 = box
            ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                        edgecolor='red', facecolor='none', linewidth=2))
            if class_names and label < len(class_names):
                ax1.text(x1, y1 - 5, class_names[label], color='red', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.5))
        ax1.axis('off')
        
        # Plot transformed image
        ax2.imshow(transformed_image)
        ax2.set_title("Transformed Image (from DataLoader)")
        for box, label in zip(transformed_boxes, transformed_labels):
            x1, y1, x2, y2 = box
            ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                        edgecolor='red', facecolor='none', linewidth=2))
            if class_names and label < len(class_names):
                ax2.text(x1, y1 - 5, class_names[label], color='red', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.5))
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        break  # Visualize only the first batch


def main():
    """Main function."""
    print("Main function started")
    args = parse_args()
    
    # Setup directories
    CFG.setup_directories()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Torch Available: ", torch.cuda.is_available())
    
    # Create data loaders
    train_dir = os.path.join(args.data_path, 'train')
    valid_dir = os.path.join(args.data_path, 'val')
    test_dir = os.path.join(args.data_path, 'test')
    # Create data loaders

    #For visualization of Original vs Transformed images
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dir, valid_dir, test_dir, batch_size=args.batch_size, shuffle = False
    )
    # Only visualize once
    if train_loader:
        #visualize_dataloader_sample(train_loader, CFG.CLASS_NAMES)
        pass
    import torchmetrics
    print(torchmetrics.__version__)
    # Main Function dilemnas
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'predict':
        predict_image(args)
    else:
        print(f"Invalid mode: {args.mode}")
    print("Main function ended")

if __name__ == "__main__":
    main()