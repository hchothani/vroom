# src/visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

def visualize_predictions(model, dataset, indices):
    """
    Visualize predictions from the model
    
    Args:
        model: PyTorch model
        dataset: PyTorch dataset
        indices: List of indices to visualize
    """
    model.eval()
    
    for idx in indices:
        image, target = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                # Add batch dimension if needed
                input_tensor = image.unsqueeze(0) if image.dim() == 3 else image
                pred = model(input_tensor)
            else:
                # Convert PIL image to tensor
                input_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)
                pred = model(input_tensor)
        
        # Visualize
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image with ground truth
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            # Denormalize if needed
            if hasattr(dataset, 'transform') and dataset.transform is not None:
                # Check if normalization was applied
                from torchvision.transforms import Normalize
                for t in dataset.transform.transforms:
                    if isinstance(t, Normalize):
                        image_np = image_np * np.array(t.std) + np.array(t.mean)
            image_np = np.clip(image_np, 0, 1)
        else:
            # Use PIL image directly
            image_np = np.array(image) / 255.0
            
        ax[0].imshow(image_np)
        ax[0].set_title("Ground Truth")
        
        # Draw ground truth boxes
        for box, label in zip(target['boxes'], target['labels']):
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax[0].add_patch(rect)
            
            class_name = dataset.class_names[label] if hasattr(dataset, 'class_names') else f"Class {label}"
            ax[0].text(
                x_min, y_min - 5, class_name, 
                bbox=dict(facecolor='green', alpha=0.5)
            )
        
        # Predictions
        ax[1].imshow(image_np)
        ax[1].set_title("Predictions")
        
        # Draw predicted boxes
        if len(pred) > 0 and hasattr(pred[0], 'boxes'):
            # Handle different prediction formats
            if hasattr(pred[0], 'boxes') and hasattr(pred[0].boxes, 'xyxy'):
                # YOLOv8 Results format
                for box, cls, conf in zip(pred[0].boxes.xyxy, pred[0].boxes.cls, pred[0].boxes.conf):
                    x_min, y_min, x_max, y_max = box.tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    rect = patches.Rectangle(
                        (x_min, y_min), width, height, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax[1].add_patch(rect)
                    
                    class_idx = int(cls.item())
                    class_name = dataset.class_names[class_idx] if hasattr(dataset, 'class_names') else f"Class {class_idx}"
                    ax[1].text(
                        x_min, y_min - 5, f"{class_name} {conf:.2f}", 
                        bbox=dict(facecolor='red', alpha=0.5)
                    )
            else:
                # Generic format
                boxes = pred[0]['boxes'] if isinstance(pred[0], dict) else pred[0].boxes
                labels = pred[0]['labels'] if isinstance(pred[0], dict) else pred[0].cls
                scores = pred[0]['scores'] if isinstance(pred[0], dict) and 'scores' in pred[0] else torch.ones_like(labels)
                
                for box, label, score in zip(boxes, labels, scores):
                    if isinstance(box, torch.Tensor):
                        x_min, y_min, x_max, y_max = box.tolist()
                    else:
                        x_min, y_min, x_max, y_max = box
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    rect = patches.Rectangle(
                        (x_min, y_min), width, height, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax[1].add_patch(rect)
                    
                    class_idx = int(label) if isinstance(label, torch.Tensor) else label
                    class_name = dataset.class_names[class_idx] if hasattr(dataset, 'class_names') else f"Class {class_idx}"
                    score_val = float(score) if isinstance(score, torch.Tensor) else score
                    ax[1].text(
                        x_min, y_min - 5, f"{class_name} {score_val:.2f}", 
                        bbox=dict(facecolor='red', alpha=0.5)
                    )
        
        plt.tight_layout()
        plt.show()

def analyze_forgetting(results):
    """
    Analyze catastrophic forgetting across tasks
    
    Args:
        results: List of evaluation results from continual learning
    """
    task_accuracies = []
    
    # Extract accuracies for each task after each training experience
    for i, result in enumerate(results):
        task_acc = []
        for j in range(i+1):
            # Try different possible metric names based on Avalanche's naming convention
            metric_names = [
                f'Top1_Acc_Exp{j}/eval_phase/test_stream/Task{j}',
                f'Top1_Acc_On_Exp{j}',
                f'Top1_Acc/eval_phase/test_stream/Task{j}',
                f'Top1_Acc_Stream/eval_phase/test_stream/Task{j}'
            ]
            
            acc = None
            for metric_name in metric_names:
                if metric_name in result:
                    acc = result[metric_name]
                    break
            
            if acc is None:
                # If we can't find the exact metric, look for any accuracy metric for this task
                for key, value in result.items():
                    if f'Task{j}' in key and 'Acc' in key:
                        acc = value
                        break
            
            if acc is None:
                # If still not found, use a placeholder
                acc = 0.0
                print(f"Warning: Could not find accuracy metric for Task {j} after training on Task {i}")
            
            task_acc.append(acc)
        
        task_accuracies.append(task_acc)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    for i, accs in enumerate(task_accuracies):
        plt.plot(range(len(accs)), accs, marker='o', label=f'After Task {i+1}')
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Task Accuracy After Each Training Experience')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot forgetting
    if len(task_accuracies) > 1:
        plt.figure(figsize=(10, 6))
        
        for j in range(len(task_accuracies[0])):
            forgetting = []
            for i in range(1, len(task_accuracies)):
                if j < len(task_accuracies[i-1]) and j < len(task_accuracies[i]):
                    # Forgetting is the difference in accuracy for task j after training on task i vs task i-1
                    forgetting.append(task_accuracies[i-1][j] - task_accuracies[i][j])
                else:
                    forgetting.append(0)
            
            plt.plot(range(1, len(task_accuracies)), forgetting, marker='s', label=f'Task {j+1}')
        
        plt.xlabel('After Training Task')
        plt.ylabel('Forgetting (decrease in accuracy)')
        plt.title('Catastrophic Forgetting Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_class_distribution(dataset):
    """
    Visualize the class distribution in a dataset
    
    Args:
        dataset: PyTorch dataset
    """
    class_counts = {}
    
    for i in range(len(dataset)):
        _, target = dataset[i]
        for label in target['labels']:
            class_idx = int(label.item())
            if class_idx in class_counts:
                class_counts[class_idx] += 1
            else:
                class_counts[class_idx] = 1
    
    # Sort by class index
    sorted_counts = {k: class_counts[k] for k in sorted(class_counts.keys())}
    
    # Get class names if available
    if hasattr(dataset, 'class_names'):
        class_names = {i: name for i, name in enumerate(dataset.class_names)}
    else:
        class_names = {i: f"Class {i}" for i in sorted_counts.keys()}
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [class_names[i] for i in sorted_counts.keys()], 
        list(sorted_counts.values())
    )
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            str(int(height)),
            ha='center', va='bottom'
        )
    
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
