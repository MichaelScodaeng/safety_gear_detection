# Safety Gear Detection System

A comprehensive implementation of safety gear detection using various RCNN architectures:
- R-CNN (Region-based Convolutional Neural Network)
- Fast R-CNN
- Faster R-CNN

This system detects safety equipment in construction sites, including:
- Hardhats (and missing hardhats)
- Masks (and missing masks)
- Safety vests (and missing safety vests)
- People, machinery, vehicles, and safety cones

## Project Structure

```
safety_gear_detection/
│
├── config.py               # Configuration settings
├── data/
│   ├── __init__.py
│   └── dataset.py          # Dataset implementation
│
├── models/
│   ├── __init__.py
│   ├── base.py             # Base RCNN implementation
│   ├── rcnn.py             # R-CNN implementation
│   ├── fast_rcnn.py        # Fast R-CNN implementation
│   └── faster_rcnn.py      # Faster R-CNN implementation
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Visualization utilities
│   └── evaluation.py       # Evaluation utilities
│
├── detector.py             # Main detector class
├── main.py                 # Entry point for training and inference
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- torchvision
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- seaborn

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/safety-gear-detection.git
   cd safety-gear-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The system expects the dataset to be organized as follows:

```
css-data/
├── train/
│   ├── images/             # Training images
│   └── labels/             # Training labels (YOLO format)
│
├── val/
│   ├── images/             # Validation images
│   └── labels/             # Validation labels (YOLO format)
│
└── test/
    ├── images/             # Test images
    └── labels/             # Test labels (YOLO format)
```

## Usage

### Training a Model

```bash
python main.py --mode train --model_type faster_rcnn --epochs 30 --batch_size 4
```

Options:
- `--model_type`: Type of model to use (`rcnn`, `fast_rcnn`, or `faster_rcnn`)
- `--data_path`: Path to the dataset directory (default: `./css-data`)
- `--epochs`: Number of epochs for training
- `--batch_size`: Batch size for training
- `--fine_tune`: Flag to fine-tune an existing model (for faster_rcnn only)
- `--freeze_backbone`: Flag to freeze backbone during fine-tuning (for faster_rcnn only)

### Evaluating a Model

```bash
python main.py --mode evaluate --model_type faster_rcnn --model_path ./models/faster_rcnn_safety_gear.pt
```

Options:
- `--model_type`: Type of model to use
- `--model_path`: Path to the saved model
- `--data_path`: Path to the dataset directory
- `--batch_size`: Batch size for evaluation

### Running Inference on an Image

```bash
python main.py --mode predict --model_type faster_rcnn --model_path ./models/faster_rcnn_safety_gear.pt --image_path ./test_image.jpg
```

Options:
- `--model_type`: Type of model to use
- `--model_path`: Path to the saved model
- `--image_path`: Path to the image for prediction
- `--confidence_threshold`: Confidence threshold for predictions
- `--nms_threshold`: NMS threshold for predictions

## Using the Detector API

You can also use the SafetyGearDetector class directly in your Python code:

```python
from detector import SafetyGearDetector

# Create a detector
detector = SafetyGearDetector(model_type='faster_rcnn')

# Load a pre-trained model
detector.load_model('./models/faster_rcnn_safety_gear.pt')

# Run inference on an image
boxes, labels, scores, class_names = detector.predict(
    'test_image.jpg',
    confidence_threshold=0.5,
    nms_threshold=0.3
)

# Visualize the results
detector.visualize_prediction('test_image.jpg', boxes, labels, scores)
```

## Model Comparison

### R-CNN
- **Pros**: The original region-based CNN approach
- **Cons**: Slow training and inference due to independent feature extraction for each region proposal

### Fast R-CNN
- **Pros**: Improved speed by sharing feature computation across region proposals
- **Cons**: Still relies on external region proposal methods

### Faster R-CNN
- **Pros**: End-to-end trainable network with integrated Region Proposal Network (RPN)
- **Cons**: More complex architecture with more parameters

## Performance Metrics

The system reports the following metrics:
- mean Average Precision (mAP)
- Precision, Recall, and F1 score per class
- Confusion matrix

## License

This project is released under the MIT License.

## Acknowledgments

- The implementation builds upon the PyTorch and torchvision libraries
- The R-CNN family of algorithms was pioneered by Ross Girshick et al.
