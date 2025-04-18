import torch
from ultralytics import YOLO

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model_path=None, num_classes=6):
        super().__init__()
        
        # Initialize YOLO model
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Start with a pretrained model
            self.model = YOLO('yolov8n.pt')
        
        # Define class names
        self.class_names = [
            "Car", "Threewheel", "Bus", "Truck", "Motorbike", "Van"
        ][:num_classes]
        
        # Update model's class information
        # Instead of directly setting names, we'll use the proper method
        # or keep track of names separately if no setter is available
        self.num_classes = num_classes
        
        # Set training mode
        self.training = True
    
    def forward(self, x, targets=None):
        # Check if x is a batch from DataLoader or a single tensor
        if isinstance(x, list):
            # Handle batch from DataLoader
            images = torch.stack([item[0] if isinstance(item[0], torch.Tensor) else self.transform(item[0]) for item in x])
            if self.training and targets is not None:
                batch_targets = [item[1] for item in x]
                formatted_targets = self._format_targets(batch_targets)
                results = self.model.model.loss(images, formatted_targets)
                return results  # This is a loss dict
            else:
                results = self.model(images)
                return results  # This is a list of Results objects
        else:
            # Handle single tensor input
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            if self.training and targets is not None:
                formatted_targets = self._format_targets([targets])
                results = self.model.model.loss(x, formatted_targets)
                return results  # This is a loss dict
            else:
                results = self.model(x)
                return results  # This is a list of Results objects
    
    def _format_targets(self, targets):
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
            return torch.zeros((0, 6), device=self.model.device if hasattr(self.model, 'device') else 'cpu')
    
    def train(self, mode=True):
        self.training = mode
        if hasattr(self.model, 'train'):
            self.model.train = mode
        return self
    
    def eval(self):
        self.training = False
        if hasattr(self.model, 'train'):
            self.model.train = False
        return self
    
    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self

