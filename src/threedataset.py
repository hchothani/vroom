import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None, class_names=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.class_names = class_names or ["Car", "Threewheel", "Bus", "Truck", "Motorbike", "Van"]
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Initialize transforms with albumentations
        self.transform = transform or A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        
        # Preload targets with original image sizes
        self.targets = []
        self._preload_targets()

    def _preload_targets(self):
        for img_file in self.image_files:
            img_path = os.path.join(self.image_dir, img_file)
            with Image.open(img_path) as img:
                orig_width, orig_height = img.size
            
            label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.txt')
            boxes, labels = self._parse_label_file(label_path)
            
            self.targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'orig_size': torch.tensor([orig_height, orig_width], dtype=torch.int64)  # H x W
            })

    def _parse_label_file(self, label_path):
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:5])
                    
                    # Convert from YOLO format to Pascal VOC
                    x_min = x_center - width/2
                    y_min = y_center - height/2
                    x_max = x_center + width/2
                    y_max = y_center + height/2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        return boxes, labels

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        
        target = self.targets[idx].copy()  # Copy to avoid modifying original
        
        # Apply albumentations transforms
        transformed = self.transform(
            image=np.array(image),
            bboxes=target['boxes'].numpy(),
            labels=target['labels'].numpy()
        )
        
        # Convert to tensor and update target
        image_tensor = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        
        # Maintain original size for YOLO loss calculation
        target['orig_size'] = target['orig_size'].clone()
        
        return image_tensor, target

    def __len__(self):
        return len(self.image_files)

    def visualize_item(self, idx):
        image, target = self[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        # Scale boxes back to original image size
        orig_h, orig_w = target['orig_size'].tolist()
        img_h, img_w = image.shape[:2]
        w_scale = orig_w / img_w
        h_scale = orig_h / img_h
        
        for box, label in zip(target['boxes'], target['labels']):
            # Scale boxes to original image dimensions
            x_min = box[0] * w_scale
            y_min = box[1] * h_scale
            x_max = box[2] * w_scale
            y_max = box[3] * h_scale
            
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min, y_min - 5, self.class_names[label],
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        plt.axis('off')
        plt.show()

def get_dataset_paths(base_path):
    return {
        'train_img_dir': os.path.join(base_path, 'train', 'images'),
        'train_label_dir': os.path.join(base_path, 'train', 'labels'),
        'val_img_dir': os.path.join(base_path, 'valid', 'images'),
        'val_label_dir': os.path.join(base_path, 'valid', 'labels')
    }

def create_dataloaders(dataset_paths, batch_size=8):
    # Use default albumentations transforms
    train_dataset = YOLODataset(
        dataset_paths['train_img_dir'],
        dataset_paths['train_label_dir']
    )

    val_dataset = YOLODataset(
        dataset_paths['val_img_dir'],
        dataset_paths['val_label_dir']
    )
    
    def collate_fn(batch):
        images = []
        targets = []
        for img, tgt in batch:
            images.append(img)
            targets.append({
                'boxes': tgt['boxes'],
                'labels': tgt['labels'],
                'orig_size': tgt['orig_size']
            })
        return torch.stack(images), targets
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

def verify_dataset(dataset_path):
    paths = get_dataset_paths(dataset_path)
    for key in ['train_img_dir', 'train_label_dir', 'val_img_dir', 'val_label_dir']:
        if not os.path.exists(paths[key]):
            print(f"Missing directory: {paths[key]}")
            return False
    
    # Test sample loading
    try:
        train_loader, _, train_dataset, _ = create_dataloaders(paths, batch_size=2)
        sample = next(iter(train_loader))
        print("Sample batch shapes:")
        print(f"Images: {sample[0].shape}")
        print(f"Boxes: {sample[1][0]['boxes'].shape}")
        train_dataset.visualize_item(0)
        return True
    except Exception as e:
        print(f"Dataset verification failed: {str(e)}")
        return False
