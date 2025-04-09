import unittest
import sys
from unittest.mock import patch, MagicMock
import torch
from io import StringIO
from ..faster_rcnn import FasterRCNN_Model

"""
Tests for the Faster R-CNN model implementation.
"""

# Import the FasterRCNN_Model class using relative import

class TestFasterRCNNModel(unittest.TestCase):
    """Test cases for FasterRCNN_Model class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock configuration with minimum required fields
        self.config = {
            'model_type': 'fasterrcnn_resnet50_fpn',
            'CLASS_NAMES': ['person', 'helmet', 'vest']
        }
        self.device = 'cpu'
        self.num_classes = 3

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_hyperparameters_basic(self, mock_stdout):
        """Test that print_hyperparameters runs without errors with minimal args"""
        # Create a minimal model with mocked components to avoid full initialization
        model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        
        # Mock internal model to avoid actual model creation
        model.model = MagicMock()
        model.model.parameters.return_value = [torch.ones(10, 10)]
        model.model.named_parameters.return_value = [('layer.weight', torch.ones(10, 10))]
        model.model.rpn = MagicMock()
        model.model.rpn.anchor_generator.sizes = [(32,), (64,), (128,)]
        model.model.rpn.anchor_generator.aspect_ratios = [(0.5, 1.0, 2.0)] * 3
        model.model.rpn.fg_iou_thresh = 0.7
        model.model.rpn.bg_iou_thresh = 0.3
        model.model.roi_heads = MagicMock()
        model.model.roi_heads.score_thresh = 0.05
        model.model.roi_heads.nms_thresh = 0.5
        model.model.roi_heads.detections_per_img = 100
        model.model.transform = MagicMock()
        model.model.transform.min_size = 640
        model.model.transform.max_size = 640
        
        # Call the method
        model.print_hyperparameters()
        
        # Check that output contains key sections
        output = mock_stdout.getvalue()
        self.assertIn("FASTER R-CNN MODEL CONFIGURATION", output)
        self.assertIn("Model Type", output)
        self.assertIn("Number of Classes", output)
        self.assertIn("Device", output)
        self.assertIn("Total Parameters", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_hyperparameters_with_training_args(self, mock_stdout):
        """Test print_hyperparameters with training arguments"""
        # Create model with mocked components
        model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        
        # Mock model attributes
        model.model = MagicMock()
        model.model.parameters.return_value = [torch.ones(10, 10)]
        model.model.named_parameters.return_value = [('layer.weight', torch.ones(10, 10))]
        model.model.rpn = MagicMock()
        model.model.roi_heads = MagicMock()
        model.model.transform = MagicMock()
        
        # Create training arguments
        training_args = {
            'epochs': 10,
            'batch_size': 4,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'gradient_accumulation_steps': 1,
            'fine_tune': True,
            'freeze_backbone': True,
            'unfreeze_layers': ['layer4', 'fpn'],
            'use_amp': True,
            'optimizer': torch.optim.Adam([torch.nn.Parameter(torch.ones(5, 5))], lr=0.001)
        }
        
        # Call the method with training arguments
        model.print_hyperparameters(training_args)
        
        # Check that output contains training configuration
        output = mock_stdout.getvalue()
        self.assertIn("TRAINING CONFIGURATION", output)
        self.assertIn("Epochs", output)
        self.assertIn("Batch Size", output)
        self.assertIn("Learning Rate", output)
        self.assertIn("Parameter Groups", output)

    @patch('sys.stdout', new_callable=StringIO)
    @patch('torch.cuda')
    def test_print_hyperparameters_with_cuda(self, mock_cuda, mock_stdout):
        """Test print_hyperparameters with CUDA information"""
        # Mock CUDA availability
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_name.return_value = "Test GPU"
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        mock_cuda.memory_reserved.return_value = 1024 * 1024 * 200  # 200MB
        mock_cuda.max_memory_allocated.return_value = 1024 * 1024 * 150  # 150MB
        
        # Create model with mocked components
        model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        model.model = MagicMock()
        model.model.parameters.return_value = [torch.ones(10, 10)]
        model.model.named_parameters.return_value = [('layer.weight', torch.ones(10, 10))]
        model.model.rpn = MagicMock()
        model.model.roi_heads = MagicMock()
        model.model.transform = MagicMock()
        
        # Call the method
        model.print_hyperparameters()
        
        # Check that output contains CUDA information
        output = mock_stdout.getvalue()
        self.assertIn("CUDA MEMORY USAGE", output)
        self.assertIn("Test GPU", output)
        self.assertIn("Memory Allocated", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_hyperparameters_handles_missing_attributes(self, mock_stdout):
        """Test that print_hyperparameters handles missing model attributes gracefully"""
        # Create model with minimal mocked components
        model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        
        # Mock model with missing components to test error handling
        model.model = MagicMock()
        model.model.parameters.return_value = [torch.ones(10, 10)]
        model.model.named_parameters.return_value = [('layer.weight', torch.ones(10, 10))]
        
        # No rpn, roi_heads, or transform attributes - should handle gracefully
        
        # Call the method
        model.print_hyperparameters()
        
        # Check that output still contains basic information
        output = mock_stdout.getvalue()
        self.assertIn("FASTER R-CNN MODEL CONFIGURATION", output)
        self.assertIn("Model Type", output)
        self.assertIn("Could not extract", output)  # Error message

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_hyperparameters_without_tabulate(self, mock_stdout):
        """Test print_hyperparameters falls back when tabulate is not available"""
        # Create model with mocked components
        model = FasterRCNN_Model(self.num_classes, self.device, self.config)
        model.model = MagicMock()
        model.model.parameters.return_value = [torch.ones(10, 10)]
        model.model.named_parameters.return_value = [('layer.weight', torch.ones(10, 10))]
        model.model.rpn = MagicMock()
        model.model.roi_heads = MagicMock()
        model.model.transform = MagicMock()
        
        # Mock imports to simulate tabulate not being available
        with patch.dict('sys.modules', {'tabulate': None}):
            # Force reload to recognize the mocked import
            if 'tabulate' in sys.modules:
                del sys.modules['tabulate']
                
            # Call the method
            model.print_hyperparameters()
            
            # Check that output still contains key information in simple format
            output = mock_stdout.getvalue()
            self.assertIn("FASTER R-CNN MODEL CONFIGURATION", output)
            self.assertIn("Model Type", output)

if __name__ == '__main__':
    unittest.main()