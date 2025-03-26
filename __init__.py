"""
Safety Gear Detection System
===========================

A comprehensive implementation of safety gear detection using various
RCNN architectures including R-CNN, Fast R-CNN, and Faster R-CNN.
"""

__version__ = '1.0.0'
__author__ = 'Safety Gear Detection Team'

from config import CFG
from data.dataset import SafetyGearDataset, get_transforms, create_data_loaders
from models.rcnn import RCNN
from models.fast_rcnn import FastRCNN 
from models.faster_rcnn import FasterRCNN_Model
from detector import SafetyGearDetector

# Setup directories on import
CFG.setup_directories()
