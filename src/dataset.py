import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import random

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None, class_names=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.class_names = class_names if class_names else [
            "Car", "Threewheel", "Bus", "Truck", "Motorbike", "Van"
        ]
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        if self.transform is None:
            image = image.resize((self.img_size, self.img_size))
        else:
            image = self.transform(image)
        
        label_file = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.txt')
        boxes = []
        labels = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:5])
                    x_min = (x_center - width/2) * orig_width
                    y_min = (y_center - height/2) * orig_height
                    x_max = (x_center + width/2) * orig_width
                    y_max = (y_center + height/2) * orig_height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_height, orig_width]),
        }
        
        return image, target
    
    def visualize_item(self, idx):
        image, target = self[idx]
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            if self.transform and any(isinstance(t, transforms.Normalize) for t in self.transform.transforms):
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        for box, label in zip(target['boxes'], target['labels']):
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            class_name = self.class_names[label]
            ax.text(
                x_min, y_min - 5, class_name, 
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

def filter_dataset_by_classes(dataset, class_indices):
    if isinstance(dataset, Subset):
        original_indices = dataset.indices
        original_dataset = dataset.dataset
        filtered_indices = []
        
        for i, idx in enumerate(original_indices):
            _, target = dataset[i]
            if any(label.item() in class_indices for label in target['labels']):
                filtered_indices.append(idx)
        
        return Subset(original_dataset, filtered_indices)
    else:
        filtered_indices = []
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            if any(label.item() in class_indices for label in target['labels']):
                filtered_indices.append(idx)
        
        return Subset(dataset, filtered_indices)

def create_dataloaders(dataset_paths, transform, batch_size=8):
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
    
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

def verify_dataset(dataset_path):
    train_dir = os.path.join(dataset_path, 'train')
    valid_dir = os.path.join(dataset_path, 'valid')
    
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        print(f"Error: Train or valid directory not found in {dataset_path}")
        return False
    
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    valid_img_dir = os.path.join(valid_dir, 'images')
    valid_label_dir = os.path.join(valid_dir, 'labels')
    
    dirs_to_check = [train_img_dir, train_label_dir, valid_img_dir, valid_label_dir]
    for dir_path in dirs_to_check:
        if not os.path.exists(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            return False
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    valid_images = [f for f in os.listdir(valid_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(train_images) == 0:
        print(f"Error: No images found in {train_img_dir}")
        return False
    
    if len(valid_images) == 0:
        print(f"Error: No images found in {valid_img_dir}")
        return False
    
    print(f"Dataset verified successfully:")
    print(f"  - Training images: {len(train_images)}")
    print(f"  - Validation images: {len(valid_images)}")
    
    random.seed(42)
    sample_train = random.sample(train_images, min(5, len(train_images)))
    sample_valid = random.sample(valid_images, min(5, len(valid_images)))
    
    missing_labels = 0
    for img in sample_train:
        label_file = os.path.join(train_label_dir, os.path.splitext(img)[0] + '.txt')
        if not os.path.exists(label_file):
            missing_labels += 1
    
    for img in sample_valid:
        label_file = os.path.join(valid_label_dir, os.path.splitext(img)[0] + '.txt')
        if not os.path.exists(label_file):
            missing_labels += 1
    
    if missing_labels > 0:
        print(f"Warning: {missing_labels} out of {len(sample_train) + len(sample_valid)} sampled images are missing label files")
    
    return True
