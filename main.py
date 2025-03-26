"""
Main script for the Safety Gear Detection System.
"""

import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import CFG
from data.dataset import SafetyGearDataset, get_transforms, create_data_loaders
from detector import SafetyGearDetector


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Safety Gear Detection System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Operation mode (train, evaluate, predict)')
    parser.add_argument('--model_type', type=str, default='faster_rcnn', 
                        choices=['rcnn', 'fast_rcnn', 'faster_rcnn'],
                        help='Model type (rcnn, fast_rcnn, faster_rcnn)')
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
                        help='Fine-tune an existing model (for faster_rcnn only)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone during fine-tuning (for faster_rcnn only)')
    parser.add_argument('--confidence_threshold', type=float, default=CFG.CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for predictions')
    parser.add_argument('--nms_threshold', type=float, default=CFG.NMS_THRESHOLD,
                        help='NMS threshold for predictions')
    parser.add_argument('--output_dir', type=str, default=CFG.OUTPUT_PATH,
                        help='Directory to save outputs')
    return parser.parse_args()


def train_model(args):
    """Train a new model or fine-tune an existing one."""
    print(f"Training {args.model_type} model...")
    
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


def main():
    """Main function."""
    args = parse_args()
    
    # Setup directories
    CFG.setup_directories()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'predict':
        predict_image(args)
    else:
        print(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()model_type=args.model_type)
    
    # Create data loaders
    train_dir = os.path.join(args.data_path, 'train')
    valid_dir = os.path.join(args.data_path, 'val')
    test_dir = os.path.join(args.data_path, 'test')
    
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dir, valid_dir, test_dir, batch_size=args.batch_size
    )
    
    if train_loader is None:
        print("No training data found. Please check the data path.")
        return None
    
    # Load model if fine-tuning
    if args.fine_tune and args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path} for fine-tuning")
        detector.load_model(args.model_path)
    
    # Train the model
    history = detector.train(
        train_loader,
        valid_loader,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY,
        fine_tune=args.fine_tune,
        freeze_backbone=args.freeze_backbone
    )
    
    # Save the model
    model_path = f"{args.output_dir}/{args.model_type}_safety_gear.pt"
    detector.save_model(model_path)
    
    # Plot training results
    if history.get('val_map') is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_map'], label='Val mAP')
        plt.title('mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{args.model_type}_training_history.png")
        plt.show()
    
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
    detector = SafetyGearDetector(