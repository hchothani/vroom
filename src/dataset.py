import os
import shutil
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from torchvision import transforms

class YOLODataset(Dataset):
    """
    Custom PyTorch Dataset for YOLO format data.
    
    Args:
        image_dir (str): Directory containing images
        label_dir (str): Directory containing label files
        img_size (int): Size to resize images to
        transform: PyTorch transforms to apply to images
        class_names (list): List of class names
    """
    def __init__(self, image_dir, label_dir, img_size=640, transform=None, class_names=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        
        # Define class names
        self.class_names = class_names if class_names else [
            "Car", "Threewheel", "Bus", "Truck", "Motorbike", "Van"
        ]
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # Resize image
        if self.transform:
            image = self.transform(image)
        
        # Load labels (YOLO format)
        label_file = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.txt')
        boxes = []
        labels = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])
                    # YOLO format: [class_id, x_center, y_center, width, height]
                    # Convert to [x_min, y_min, x_max, y_max]
                    x_center, y_center, width, height = map(float, data[1:5])
                    
                    # Convert normalized coordinates to absolute
                    x_min = (x_center - width/2) * orig_width
                    y_min = (y_center - height/2) * orig_height
                    x_max = (x_center + width/2) * orig_width
                    y_max = (y_center + height/2) * orig_height
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_height, orig_width]),
        }
        
        return image, target
    
    def visualize_item(self, idx):
        """Visualize an item from the dataset with bounding boxes"""
        image, target = self[idx]
        
        # Convert image tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            # Denormalize if needed
            if self.transform and any(isinstance(t, transforms.Normalize) for t in self.transform.transforms):
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        # Draw bounding boxes
        for box, label in zip(target['boxes'], target['labels']):
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = self.class_names[label]
            ax.text(
                x_min, y_min - 5, class_name, 
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        plt.axis('off')
        plt.show()


def organize_dataset(base_path, train_ratio=0.7):
    """
    Organize the dataset into train and validation sets
    
    Args:
        base_path (str): Path to the dataset directory
        train_ratio (float): Ratio of training data (0-1)
        
    Returns:
        dict: Dictionary with paths to train and validation data
    """
    # Create directories
    os.makedirs(os.path.join(base_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val', 'labels'), exist_ok=True)
    
    # Get all image files
    image_dir = os.path.join(base_path, 'images')
    label_dir = os.path.join(base_path, 'labels')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(
        image_files, train_size=train_ratio, random_state=42
    )
    
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    # Move files to respective directories
    for file in train_files:
        # Copy images
        shutil.copy(
            os.path.join(image_dir, file),
            os.path.join(base_path, 'train', 'images', file)
        )
        
        # Copy labels if they exist
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.copy(
                os.path.join(label_dir, label_file),
                os.path.join(base_path, 'train', 'labels', label_file)
            )
    
    for file in val_files:
        # Copy images
        shutil.copy(
            os.path.join(image_dir, file),
            os.path.join(base_path, 'val', 'images', file)
        )
        
        # Copy labels if they exist
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.copy(
                os.path.join(label_dir, label_file),
                os.path.join(base_path, 'val', 'labels', label_file)
            )
    
    return {
        'train_img_dir': os.path.join(base_path, 'train', 'images'),
        'train_label_dir': os.path.join(base_path, 'train', 'labels'),
        'val_img_dir': os.path.join(base_path, 'val', 'images'),
        'val_label_dir': os.path.join(base_path, 'val', 'labels')
    }


def filter_dataset_by_classes(dataset, class_indices):
    """
    Filter dataset to only include specific classes
    
    Args:
        dataset: PyTorch dataset to filter
        class_indices (list): List of class indices to include
        
    Returns:
        Subset: Filtered dataset
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if any(label.item() in class_indices for label in target['labels']):
            filtered_indices.append(idx)
    
    return torch.utils.data.Subset(dataset, filtered_indices)


def create_dataloaders(dataset_paths, transform, batch_size=8):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        dataset_paths (dict): Dictionary with dataset paths
        transform: PyTorch transforms
        batch_size (int): Batch size for DataLoader
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = YOLODataset(
        dataset_paths['train_img_dir'],
        dataset_paths['train_label_dir'],
        transform=transform
    )

    val_dataset = YOLODataset(
        dataset_paths['val_img_dir'],
        dataset_paths['val_label_dir'],
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: x  # Custom collate function to handle variable-sized data
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: x
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

