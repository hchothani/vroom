import torch
from ultralytics import YOLO

class YOLOWrapper(torch.nn.Module):
    """
    Wrapper around YOLOv8 to make it compatible with Avalanche
    
    Args:
        model_path (str, optional): Path to a pretrained YOLO model
        num_classes (int): Number of classes to detect
    """
    def __init__(self, model_path=None, num_classes=6):
        super().__init__()
        
        # Initialize YOLOv8 model
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Start with a pretrained model and modify the detection head
            self.model = YOLO('yolov8n.pt')
            # Update the model's class count
            self.model.model.names = {i: name for i, name in enumerate([
                "Car", "Threewheel", "Bus", "Truck", "Motorbike", "Van"
            ])}
            self.model.model.nc = num_classes
    
    def forward(self, x, targets=None):
        """
        Forward pass for the model
        
        Args:
            x: Input tensor or batch of tensors
            targets: Optional target tensors for training
            
        Returns:
            Model output (loss during training, predictions during inference)
        """
        # Check if x is a batch from DataLoader or a single tensor
        if isinstance(x, list):
            # Handle batch from DataLoader
            images = torch.stack([item[0] for item in x])
            if self.training and targets is not None:
                # Format targets for YOLOv8
                formatted_targets = self._format_targets([item[1] for item in x])
                loss_dict = self.model.model.loss(images, formatted_targets)
                return loss_dict
            else:
                # Inference mode
                return self.model(images)
        else:
            # Handle single tensor input
            if self.training and targets is not None:
                # Format targets for YOLOv8
                formatted_targets = self._format_targets([targets])
                loss_dict = self.model.model.loss(x.unsqueeze(0), formatted_targets)
                return loss_dict
            else:
                # Inference mode
                return self.model(x.unsqueeze(0) if x.dim() == 3 else x)
    
    def _format_targets(self, targets):
        """
        Format PyTorch targets to YOLOv8 format
        
        Args:
            targets: List of target dictionaries
            
        Returns:
            Formatted targets compatible with YOLOv8
        """
        # This implementation depends on YOLOv8's expected format
        # For YOLOv8, we need to convert to a list of tensors with the format:
        # [image_idx, class_id, x, y, w, h]
        
        formatted_targets = []
        
        for img_idx, target in enumerate(targets):
            boxes = target['boxes']
            labels = target['labels']
            
            if len(boxes) == 0:
                continue
                
            # Get original image dimensions
            orig_h, orig_w = target['orig_size']
            
            # Convert absolute coordinates [x1, y1, x2, y2] to normalized [x_center, y_center, width, height]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # Calculate centers and dimensions
            x_center = (x1 + x2) / 2.0 / orig_w
            y_center = (y1 + y2) / 2.0 / orig_h
            width = (x2 - x1) / orig_w
            height = (y2 - y1) / orig_h
            
            # Stack into tensor [img_idx, class_id, x_center, y_center, width, height]
            img_targets = torch.stack([
                torch.full_like(labels, img_idx, dtype=torch.float32),
                labels.float(),
                x_center,
                y_center,
                width,
                height
            ], dim=1)
            
            formatted_targets.append(img_targets)
        
        if formatted_targets:
            return torch.cat(formatted_targets, dim=0)
        else:
            # Return empty tensor with correct shape if no targets
            return torch.zeros((0, 6))
    
    def get_parameters(self):
        """Get model parameters for optimization"""
        return self.model.parameters()
