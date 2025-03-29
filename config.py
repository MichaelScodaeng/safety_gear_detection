"""
Configuration settings for the Safety Gear Detection System.
"""

import os
import torch

class CFG:
    """Configuration class for safety gear detection system."""
    
    DEBUG = True
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 42

    TARGET_RATIOS = {'train': 0.80, 'valid': 0.15, 'test': 0.05}

    # Class map from dataset description
    PPE_CLASSES = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "Machinery",
        9: "Vehicle"
    }
    NUM_CLASSES = len(PPE_CLASSES)
    CLASS_NAMES = list(PPE_CLASSES.values())
    # Training
    EPOCHS = 2 if DEBUG else 30
    BATCH_SIZE = 4 if DEBUG else 12
    IMGSZ = 640  # Resize to 640x640
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.9
    
    # Image Processing
    RECT = False  # or True -> but makes shuffle = false due incompatibility
    VERBOSE = False  # Suppress detailed logs
    SHOW_LABELS = False  # Hide labels in console output
    SHOW_CONF = False  # Hide confidence scores in output
    SHOW_BOXES = False  # Hide bounding boxes in terminal output
    
    # Detection
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.3
    
    # Model
    BASE_MODEL = 'yolo11n.pt'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    CSS_DATA_PATH = './css-data'
    OUTPUT_PATH = './output'
    WORKING_PATH = "./working"
    MODEL_PATH = './models'
    FOLDERS = ["train", "val", "test"]
    
    @staticmethod
    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories for the project."""
        for directory in [cls.WORKING_PATH, cls.OUTPUT_PATH, cls.MODEL_PATH]:
            cls.ensure_directory_exists(directory)
        
        # Define paths for results
        cls.TRAIN_RESULTS = f"{cls.WORKING_PATH}/train_results.csv"
        cls.MODEL_WEIGHTS_PATH = f'{cls.WORKING_PATH}/runs/detect/train/weights/best.pt'
        cls.OUTPUT_MODEL_PATH = f'{cls.WORKING_PATH}/best.pt'
        cls.METADATA_PATH = f'{cls.WORKING_PATH}/metadata.json'
