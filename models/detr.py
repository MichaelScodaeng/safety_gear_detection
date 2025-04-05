import os
import torch
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch import nn


class DETR_Model:
    """
    Wrapper for DETR (DEtection TRansformer) model from Hugging Face.
    """

    def __init__(self, num_classes, device=None, config=None):
        """
        Initialize the DETR model.

        Args:
            num_classes (int): Number of classes to detect.
            device (str): Device to use (cuda or cpu).
            config (dict): Configuration parameters.
        """
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self.model_type = self.config.get('model_type', 'facebook/detr-resnet-50')
        self.threshold = self.config.get('confidence_threshold', 0.5)
        self.model = None
        self.processor = None
        self._initialize_model()
        print(f"DETR model initialized on {self.device} with {self.num_classes} classes.")
        print(f"Model type: {self.model_type}")

    def _initialize_model(self):
        """Initialize the DETR model and processor."""
        print(f"Initializing DETR model: {self.model_type}")
        self.processor = DetrImageProcessor.from_pretrained(self.model_type)
        self.model = DetrForObjectDetection.from_pretrained(self.model_type)

        if self.num_classes != 91:
            in_features = self.model.class_labels_classifier.in_features
            self.model.class_labels_classifier = nn.Linear(in_features, self.num_classes)

            # Update config for proper loss computation
            self.model.config.num_labels = self.num_classes
            self.model.config.id2label = {i: f"class_{i}" for i in range(self.num_classes)}
            self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}

            print(f"Adjusted classification head for {self.num_classes} classes.")

        # Move the entire model to the correct device
        self.model.to(self.device)

        # Debugging: Check if all model parameters are on the correct device
        for name, param in self.model.named_parameters():
            if param.device != torch.device(self.device):
                print(f"Parameter {name} is on {param.device}, expected {self.device}")

    def predict(self, image, confidence_threshold=None):
        """
        Run inference on an image.

        Args:
            image: Image to run inference on (file path, PIL Image, or numpy array).
            confidence_threshold (float, optional): Confidence threshold.

        Returns:
            tuple: (boxes, labels, scores, class_names)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        threshold = confidence_threshold or self.threshold
        encoding = self.processor(images=image, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        class_name_map = self.config.get('CLASS_NAMES', {})
        class_names = [class_name_map.get(int(label), str(label)) for label in labels]

        return boxes, labels, scores, class_names

    def train(self, train_loader, valid_loader=None, epochs=10, lr=0.0001, weight_decay=0.0001):
        """
        Train the DETR model.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)

        history = {'train_loss': [], 'val_loss': [] if valid_loader else None}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for images, targets in train_loader:
                # Move images and targets to the correct device
                images = images.to(self.device)
                detr_targets = []
                for t in targets:
                    target_dict = {
                        'labels': t['labels'].to(self.device),
                        'boxes': t['boxes'].to(self.device)
                    }
                    detr_targets.append(target_dict)

                optimizer.zero_grad()
                outputs = self.model(pixel_values=images, labels=detr_targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

            if valid_loader:
                val_loss = self._validate(valid_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}")

            lr_scheduler.step()

        return history

    def _validate(self, valid_loader):
        """Validate the model on a validation dataset."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in valid_loader:
                images = images.to(self.device)

                detr_targets = []
                for t in targets:
                    target_dict = {
                        'class_labels': t['labels'].to(self.device),
                        'boxes': t['boxes'].to(self.device)
                    }
                    detr_targets.append(target_dict)

                outputs = self.model(pixel_values=images, labels=detr_targets)
                val_loss += outputs.loss.item()

        return val_loss / len(valid_loader)

    def save_model(self, filepath):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model_type = checkpoint['model_type']
        self.num_classes = checkpoint['num_classes']
        self.config = checkpoint['config']
        self._initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        self.model.to(self.device)