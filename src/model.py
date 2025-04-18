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
        
        for param in self.model.model.parameters():
            param.requires_grad = True

#        self.model.model.head.nc = num_classes
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
        print(targets[0])
        print("Target type:", type(targets))
        print("Target shape:", targets.shape if isinstance(targets, torch.Tensor) else "N/A")
        
#        for img_idx, target in enumerate(targets):
#            boxes = target['boxes']
#            labels = target['labels']
            
#            if len(boxes) == 0:
#                continue
            
            # Get original image dimensions
#            orig_h, orig_w = target['orig_size']
            
            # Convert absolute coordinates [x1, y1, x2, y2] to normalized [x_center, y_center, width, height]
#            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
#            x_center = (x1 + x2) / 2.0 / orig_w
#            y_center = (y1 + y2) / 2.0 / orig_h
#            width = (x2 - x1) / orig_w
#            height = (y2 - y1) / orig_h
            
            # Stack into tensor [img_idx, class_id, x_center, y_center, width, height]
#            img_targets = torch.stack([
#                torch.full_like(labels, img_idx, dtype=torch.float32),
#                labels.float(),
#                x_center,
#                y_center,
#                width,
#                height
#            ], dim=1)
#            
#            formatted_targets.append(img_targets)
        
        boxes = []
        labels = []
        for img_idx, target_tensor in enumerate(targets):
            # Extract boxes (columns 0-3), labels (column 4), orig_size (column 5)
            boxes = target_tensor[:, :4]  # Shape: [num_objects, 4]
            labels = target_tensor[:, 4].long()  # Shape: [num_objects]
            orig_size = target_tensor[0, 5]  # Assuming orig_size is same for all objects
            
            if len(boxes) == 0:
                continue

            # Get original image dimensions
            orig_h, orig_w = orig_size  # Adjust if stored differently
            
            # Convert to normalized YOLO format
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
        return torch.cat(formatted_targets, dim=0)

        

    
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
    
    def parameters(self, recurse=True):
        return self.model.model.parameters(recurse=recurse)

    def to(self, device):
        if hasattr(self.model, 'to'):
            self.model.model.to(device)
        return self

