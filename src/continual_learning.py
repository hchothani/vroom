import torch
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.strategies import Naive, CWRStar, EWC, LwF, GEM

def create_cl_benchmark(train_dataset, val_dataset, class_groups):
    """
    Create a continual learning benchmark
    
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
        task_train = filter_dataset_by_classes(train_dataset, class_group)
        task_val = filter_dataset_by_classes(val_dataset, class_group)
        
        train_datasets.append(task_train)
        val_datasets.append(task_val)
    
    # Create Avalanche benchmark
    benchmark = dataset_benchmark(
        train_datasets=train_datasets,
        test_datasets=val_datasets,
        task_labels=list(range(len(class_groups))),
        complete_test_set_only=False
    )
    
    return benchmark

def create_cl_strategy(model, strategy_name='naive', lr=0.001):
    """
    Create a continual learning strategy
    
    Args:
        model: PyTorch model
        strategy_name (str): Name of the strategy to use
        lr (float): Learning rate
        
    Returns:
        Avalanche strategy object
    """
    # Define the evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True),
        loss_metrics(epoch=True, experience=True),
        loggers=[InteractiveLogger()]
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Select strategy based on name
    if strategy_name.lower() == 'naive':
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin
        )
    elif strategy_name.lower() == 'ewc':
        # Elastic Weight Consolidation
        strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            ewc_lambda=0.4,  # Regularization strength
            train_mb_size=8,
            train_epochs=5,
            eval_mb_size=8,
            evaluator=eval_plugin
        )
    elif strategy_name.lower() == 'lwf':
        # Learning without Forgetting
        strategy = LwF(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            alpha
