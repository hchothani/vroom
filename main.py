import torch
from torchvision import transforms
from src.dataset import organize_dataset, create_dataloaders
from src.model import YOLOWrapper
from src.continual_learning import create_cl_benchmark, create_cl_strategy
from src.visualization import visualize_predictions, analyze_forgetting

# Set random seeds for reproducibility
torch.manual_seed(42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Organize dataset
dataset_paths = organize_dataset('data/vehicle_dataset')

# Create dataloaders
train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(dataset_paths, transform)

# Initialize model
model = YOLOWrapper(num_classes=6)

# Define class groups for tasks
class_groups = [[0, 1], [2, 3], [4, 5]]  # Car+Threewheel, Bus+Truck, Motorbike+Van

# Create continual learning benchmark
cl_benchmark = create_cl_benchmark(train_dataset, val_dataset, class_groups)

# Create continual learning strategy
cl_strategy = create_cl_strategy(model, strategy_name='ewc', lr=0.001)

# Train on the continual learning benchmark
print("Starting Continual Learning Training...")
results = []

for experience in cl_benchmark.train_stream:
    print(f"Start training on experience {experience.current_experience}")
    cl_strategy.train(experience)
    eval_results = cl_strategy.eval(cl_benchmark.test_stream[:experience.current_experience+1])
    results.append(eval_results)
    print(f"Finished training on experience {experience.current_experience}")

print("Training completed!")

# Evaluate the final model on all tasks
print("Final Evaluation:")
for i, experience in enumerate(cl_benchmark.test_stream):
    print(f"Evaluating on Task {i+1}:")
    cl_strategy.eval(experience)

# Visualize some predictions
test_indices = [0, 10, 20, 30, 40]
visualize_predictions(model, val_dataset, test_indices)

# Analyze forgetting
analyze_forgetting(results)
