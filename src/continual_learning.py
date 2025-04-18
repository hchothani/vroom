import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.evaluation.metrics.detection import DetectionMetrics
from avalanche.evaluation.metrics import timing_metrics, loss_metrics
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.supervised import Naive, EWC, LwF, Replay, GEM

def create_cl_benchmark(train_dataset, val_dataset, class_groups):
    """
    Create a continual learning benchmark using Avalanche
    
    Args:
        train_dataset: Full training dataset
        val_dataset: Full validation dataset
        class_groups: List of lists, where each inner list contains class indices for a task
        
    Returns:
        Avalanche benchmark object
    """
    from src.dataset import filter_dataset_by_classes
    
    # Create task-specific datasets
    train_datasets = []
    val_datasets = []
    
    for class_group in class_groups:
        task_train = AvalancheDataset(filter_dataset_by_classes(train_dataset, class_group))
        task_val = AvalancheDataset(filter_dataset_by_classes(val_dataset, class_group))
        task_train = AvalancheDataset(task_train)
        
        train_datasets.append(task_train)
        val_datasets.append(task_val)
    
    # Create Avalanche benchmark
#    d_train = {i: task_train for i, task_train in enumerate(task_train)} 
#    d_valid = {i: task_val for i, task_val in enumerate(task_val)}

    vald=AvalancheDataset(val_dataset)
    bm = benchmark_from_datasets(train_datasets=train_datasets,test_datasets=vald)

    return bm

class YOLOLoss(torch.nn.Module):
    """
    Custom loss wrapper for YOLOv8 to work with Avalanche
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        """
        Forward pass for the loss
        
        Args:
            outputs: Model outputs (already loss dict from YOLOv8)
            targets: Target tensors (not used, as loss is already computed)
            
        Returns:
            Total loss value
        """
        # YOLOv8 returns a dictionary of losses
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        
        # If it's already a tensor, return it
        if isinstance(outputs, torch.Tensor):
            return outputs
        
        # If it's a list of tensors, sum them
        if isinstance(outputs, list) and all(isinstance(x, torch.Tensor) for x in outputs):
            return sum(outputs)
        
        # Default case - should not happen
        raise ValueError(f"Unexpected output type: {type(outputs)}")

def make_ds_metrics(detention_only=True):
    if detection_only:
        iou_types = ["bbox"]
    else:
        iou_types = ["bbox", "segm"]
    return DetectionMetrics(
            iou_types=iou_types, default_to_coco=True, summarize_to_stdou=True
            )

def create_cl_strategy(model, strategy_name='naive', lr=0.001, mem_size=200):
    """
    Create a continual learning strategy using Avalanche
    
    Args:
        model: PyTorch model (YOLOWrapper)
        strategy_name (str): Name of the strategy to use
        lr (float): Learning rate
        mem_size (int): Memory size for replay-based strategies
        
    Returns:
        Avalanche strategy object
    """
    # Define the evaluation plugin with metrics and loggers
    loggers = [
        InteractiveLogger(),
        TensorboardLogger(tb_log_dir="./logs")
    ]
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True),
        timing_metrics(epoch=True),
        loss_metrics(epoch=True, experience=True),
        make_ds_metrics(detection_only=True),
        loggers=loggers,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function
    criterion = YOLOLoss()
    
    # Select strategy based on name
    strategy_name = strategy_name.lower()
    
    if strategy_name == 'naive':
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif strategy_name == 'ewc':
        # Elastic Weight Consolidation
        strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            ewc_lambda=0.4,  # Regularization strength
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif strategy_name == 'lwf':
        # Learning without Forgetting
        strategy = LwF(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            alpha=1.0,  # Distillation hyperparameter
            temperature=2.0,  # Softmax temperature for distillation
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif strategy_name == 'replay':
        # Experience Replay
        strategy = Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=mem_size,
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    elif strategy_name == 'gem':
        # Gradient Episodic Memory
        strategy = GEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            patterns_per_exp=mem_size,
            memory_strength=0.5,
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy

def get_available_strategies():
    """
    Get a list of available continual learning strategies
    
    Returns:
        list: List of available strategy names
    """
    return ['naive', 'ewc', 'lwf', 'replay', 'gem']
